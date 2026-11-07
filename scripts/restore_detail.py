#!/usr/bin/env python3
"""Apply learned local RGB mappings to a registered native-resolution original.

Requires a job-specific regions JSON or explicit subject masks; no assumed
person coordinates, model weights, network, or image-generation API calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont, ImageOps

from cv_core import blur, correlation, dense_flow, log, normalize_for_flow, save_rgb, smoothstep


def fingerprint(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_image(path):
    with Image.open(path) as opened:
        im = ImageOps.exif_transpose(opened)
        rgba = im.convert('RGBA')
        icc = im.info.get('icc_profile')
        if icc:
            import io
            rgb = ImageCms.profileToProfile(im.convert('RGB'),
                ImageCms.ImageCmsProfile(io.BytesIO(icc)), ImageCms.createProfile('sRGB'), outputMode='RGB')
            rgba = rgb.convert('RGBA')
            rgba.putalpha(im.convert('RGBA').getchannel('A'))
        return rgba.copy()


def flow_rgb(rgba):
    array = np.asarray(rgba)
    alpha = array[..., 3:4].astype(np.float32) / 255
    return np.uint8(np.rint(array[..., :3] * alpha + 127 * (1 - alpha)))


def polygon_mask(shape, polygons):
    h, w = shape
    result = np.zeros((h, w), np.uint8)
    for points in polygons:
        array = np.asarray(points, np.float32)
        if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] != 2:
            raise ValueError('Each polygon needs at least three [x,y] points')
        if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ValueError('Region coordinates must be finite and normalized to [0,1]')
        array *= [w - 1, h - 1]
        cv.fillPoly(result, [np.rint(array).astype(np.int32)], 1)
    return result


def subject_mask(rgba, spec, config_dir):
    """All geometry comes from this job. Alpha and exclusions are hard limits."""
    h, w = rgba.height, rgba.width
    arr = np.asarray(rgba)
    opaque = (arr[..., 3] >= 250).astype(np.uint8)
    spec = spec or {}
    if 'mask' in spec:
        path = Path(spec['mask'])
        if not path.is_absolute():
            path = config_dir / path
        with Image.open(path) as im:
            mask = np.array(im.convert('L').resize((w, h), Image.Resampling.NEAREST)) >= 128
        mask = mask.astype(np.uint8)
    elif spec.get('fg_polygons'):
        box = np.asarray(spec.get('box', [0, 0, 1, 1]), dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any((box < 0) | (box > 1)):
            raise ValueError('box must be normalized [left,top,right,bottom]')
        x0, y0, x1, y1 = np.rint(box * [w, h, w, h]).astype(int)
        if x1 <= x0 or y1 <= y0:
            raise ValueError('box must have positive dimensions')
        seeds = np.full((h, w), cv.GC_BGD, np.uint8)
        seeds[y0:y1, x0:x1] = cv.GC_PR_BGD
        seeds[polygon_mask((h, w), spec['fg_polygons']) > 0] = cv.GC_FGD
        background = polygon_mask((h, w), spec.get('bg_polygons', []))
        seeds[(background > 0) | (opaque == 0)] = cv.GC_BGD
        if not np.any(seeds == cv.GC_BGD) or not np.any(seeds == cv.GC_FGD):
            raise ValueError('GrabCut needs both certain foreground and certain background seeds')
        cv.grabCut(flow_rgb(rgba), seeds, None, np.zeros((1, 65)),
                   np.zeros((1, 65)), 6, cv.GC_INIT_WITH_MASK)
        mask = ((seeds == cv.GC_FGD) | (seeds == cv.GC_PR_FGD)).astype(np.uint8)
        if spec.get('fill_holes', False):
            # Padded flood fill works even if the subject touches an image corner.
            filled = np.pad(mask, 1)
            cv.floodFill(filled, None, (0, 0), 2)
            mask[filled[1:-1, 1:-1] == 0] = 1
        mask[background > 0] = 0
    elif np.any(arr[..., 3] < 250):
        mask = opaque.copy()
    else:
        raise ValueError('Provide a job-specific subject mask or foreground seed polygons for each opaque input')
    mask[polygon_mask((h, w), spec.get('exclude_polygons', [])) > 0] = 0
    mask *= opaque
    if not np.any(mask):
        raise ValueError('No protected foreground remains in this input')
    return mask


def affine(source, target, smask, tmask):
    detector = cv.SIFT_create(nfeatures=6500, contrastThreshold=0.015)
    ks, ds = detector.detectAndCompute(normalize_for_flow(source), smask * 255)
    kt, dt = detector.detectAndCompute(normalize_for_flow(target), tmask * 255)
    identity = np.eye(2, 3, dtype=np.float32)
    info = dict(accepted=False, source_keypoints=len(ks), base_keypoints=len(kt), inliers=0)
    if ds is None or dt is None or len(dt) < 2:
        return identity, info
    h, w = source.shape[:2]
    good = []
    for pair in cv.BFMatcher().knnMatch(ds, dt, k=2):
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.72 * n.distance and np.linalg.norm(np.subtract(ks[m.queryIdx].pt, kt[m.trainIdx].pt)) < 0.25 * np.hypot(w, h):
            good.append(m)
    info['ratio_matches'] = len(good)
    if len(good) < 12:
        return identity, info
    ps = np.float32([ks[m.queryIdx].pt for m in good])
    pt = np.float32([kt[m.trainIdx].pt for m in good])
    matrix, inliers = cv.estimateAffinePartial2D(ps, pt, method=cv.RANSAC,
        ransacReprojThreshold=3.5, maxIters=4000, confidence=0.999, refineIters=30)
    count = int(inliers.sum()) if inliers is not None else 0
    info['inliers'] = count
    if matrix is not None and count >= 12:
        scale = np.linalg.norm(matrix[:, 0])
        span = np.ptp(pt[inliers.ravel().astype(bool)], axis=0)
        angle = abs(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
        if 0.85 < scale < 1.18 and angle < 12 and span[0] > 0.12 * w and span[1] > 0.12 * h:
            info['accepted'] = True
            return matrix.astype(np.float32), info
    return identity, info


def analyze(args, regions):
    src, base = load_image(args.source), load_image(args.base)
    aspect_error = abs(np.log((src.width / src.height) / (base.width / base.height)))
    if aspect_error > 0.02:
        raise ValueError('Edited aspect ratio differs by more than 2%; do not stretch the original. Review the generated edit.')
    scale = min(1, args.work_long_edge / max(base.size))
    size = (max(64, round(base.width * scale)), max(64, round(base.height * scale)))
    bimg = base.resize(size, Image.Resampling.LANCZOS)
    simg = src.resize(size, Image.Resampling.LANCZOS)
    target, small = flow_rgb(bimg), flow_rgb(simg)
    log('Segmenting job-specific protected foreground in both inputs')
    tmask = subject_mask(bimg, regions.get('base'), args.regions.parent)
    smask = subject_mask(simg, regions.get('source'), args.regions.parent)
    matrix, ainfo = affine(small, target, smask, tmask)
    log('Affine matching: ' + json.dumps(ainfo))
    w, h = size
    aligned = cv.warpAffine(small, matrix, size, flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REFLECT_101)
    adist = cv.warpAffine(cv.distanceTransform(smask, cv.DIST_L2, 5), matrix, size)
    valid_a = cv.warpAffine(np.ones((h, w), np.float32), matrix, size)
    log('Checking local registration in both directions')
    first, second = normalize_for_flow(target), normalize_for_flow(aligned)
    flow, back = dense_flow(first, second), dense_flow(second, first)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    mx, my = xx + flow[..., 0], yy + flow[..., 1]
    def remap(a):
        return cv.remap(a, mx, my, cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    warped = remap(aligned)
    fb = np.linalg.norm(flow + remap(back), axis=2)
    dist = np.minimum(cv.distanceTransform(tmask, cv.DIST_L2, 5), remap(adist))
    support = smoothstep(dist, 2, 9)
    support *= (mx >= 1) & (mx < w - 2) & (my >= 1) & (my < h - 2) & (remap(valid_a) > 0.999)
    lt = cv.cvtColor(target.astype(np.float32) / 255, cv.COLOR_RGB2Lab)
    ls = cv.cvtColor(warped.astype(np.float32) / 255, cv.COLOR_RGB2Lab)
    corr = correlation(lt[..., 0], ls[..., 0])
    pt, ps = lt[..., 0] - blur(lt[..., 0], 16), ls[..., 0] - blur(ls[..., 0], 16)
    vt, vs = np.sqrt(blur(pt * pt, 9) + 2), np.sqrt(blur(ps * ps, 9) + 2)
    photo = np.exp(-np.square(blur(np.abs(pt / vt - ps / vs), 3) / 1.15))
    chroma_delta = lt[..., 1:] - ls[..., 1:]
    chroma_residual = blur(chroma_delta, 2) - blur(chroma_delta, 12)
    color = np.exp(-np.square(np.linalg.norm(chroma_residual, axis=2) / 12))
    geometric = np.exp(-np.square(fb / 2.4))
    uy, ux = np.gradient(flow[..., 0])
    vy, vx = np.gradient(flow[..., 1])
    det = (1 + ux) * (1 + vy) - uy * vx
    strain = np.sqrt(ux * ux + uy * uy + vx * vx + vy * vy)
    deformation = np.exp(-np.square(strain / 0.70)) * ((det > 0.4) & (det < 2.1))
    common = geometric * photo * color * deformation
    fine = 0.95 * support * common * smoothstep(corr, 0.25, 0.80)
    medium = 0.85 * support * common * smoothstep(corr, 0.50, 0.90)
    fine = np.minimum(blur(fine, 1.4), support * 0.95)
    medium = np.minimum(blur(medium, 1.6), support * 0.85)
    reject = (geometric < 0.1) | (deformation == 0) | (corr < 0.25)
    fine[reject] = 0
    medium[reject | (corr < 0.5)] = 0
    if not ainfo['accepted']:
        fine[:] = 0
        medium[:] = 0
        log('No reliable shared feature geometry: preserving edited base without source transfer')
    gain = blur(np.clip(vt / vs, 0.75, 1.40), 8)
    np.savez_compressed(args.out / 'registration.npz', flow=flow, matrix=matrix,
        fine=fine, medium=medium, support=support, gain=gain)
    for name, image in [('base-mask', tmask * 255), ('source-mask', smask * 255),
                        ('detail-weight', np.uint8(fine * 255))]:
        Image.fromarray(image).save(args.out / f'{name}.png')
    save_rgb(args.out / 'source-registered-preview.png', warped)
    heat = cv.cvtColor(cv.applyColorMap(np.uint8(fine / .95 * 255), cv.COLORMAP_TURBO), cv.COLOR_BGR2RGB)
    overlay = target.copy()
    inside = support > 0
    overlay[inside] = np.uint8(target[inside] * .4 + heat[inside] * .6)
    save_rgb(args.out / 'confidence-overlay.png', overlay)
    interior = support > 0.8
    report = dict(source=str(args.source.resolve()), base=str(args.base.resolve()),
        source_sha256=fingerprint(args.source), base_sha256=fingerprint(args.base),
        regions_sha256=fingerprint(args.regions), source_size=list(src.size),
        base_size=list(base.size), working_size=list(size), affine=ainfo,
        protected_interior_pixels=int(interior.sum()),
        reliable_fraction=float((fine[interior] > .25).mean()) if interior.any() else 0,
        mean_detail_weight=float(fine[interior].mean()) if interior.any() else 0,
        transfer_enabled=bool(np.any(fine > 0)), luminance_only=True,
        source_color_transfer=False, generative_restoration=False, sharpening=False)
    method=getattr(args,'method','local-rgb')
    report['restore_method']=method
    if method=='local-rgb':
        import mapping_core
        log('Learning structure-gated local monotonic RGB curves')
        fields,metadata=mapping_core.fit(np.asarray(src)[...,:3],np.asarray(base)[...,:3],
            dict(flow=flow,matrix=matrix,support=support),ainfo['accepted'],
            getattr(args,'structure_channel','auto'))
        field_path=args.out/'local-transform-field.npz'
        np.savez_compressed(field_path,**fields)
        report.update(metadata)
        report['mapping_sha256']=fingerprint(field_path)
        Image.fromarray(np.uint8(fields['gate']*255)).save(args.out/'detail-weight.png')
        heat=cv.cvtColor(cv.applyColorMap(np.uint8(fields['gate']*255),cv.COLORMAP_TURBO),cv.COLOR_BGR2RGB)
        overlay=target.copy()
        overlay[inside]=np.uint8(target[inside]*.4+heat[inside]*.6)
        save_rgb(args.out/'confidence-overlay.png',overlay)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    log('Analysis saved. Inspect masks and confidence overlay before fusion.')


def tile_data(data, width, height, top, bottom):
    h, w = data['flow'].shape[:2]
    gx = (np.arange(width, dtype=np.float32) + .5) * w / width - .5
    gy = (np.arange(top, bottom, dtype=np.float32) + .5) * h / height - .5
    mx, my = np.meshgrid(gx, gy)
    def field(name, interpolation=cv.INTER_LINEAR):
        return cv.remap(data[name], mx, my, interpolation, borderMode=cv.BORDER_REPLICATE)
    flow = field('flow', cv.INTER_CUBIC)
    ax, ay = mx + flow[..., 0], my + flow[..., 1]
    inv = cv.invertAffineTransform(data['matrix'])
    sx = (inv[0, 0] * ax + inv[0, 1] * ay + inv[0, 2] + .5) * width / w - .5
    sy = (inv[1, 0] * ax + inv[1, 1] * ay + inv[1, 2] + .5) * height / h - .5
    return sx, sy, {key: field(key) for key in ['fine', 'medium', 'support', 'gain']}


def detail_crop(name, center, original, base, result, out):
    w, h = base.size
    side = min(512, w, h)
    cx = min(max(round(center[0] * w), side // 2), w - side // 2)
    cy = min(max(round(center[1] * h), side // 2), h - side // 2)
    box = (cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side)
    board = Image.new('RGB', (3 * side + 72, side + 158), '#eeeeee')
    draw = ImageDraw.Draw(board)
    draw.text((18, 15), name + ': native-scale detail', fill='#111111', font=ImageFont.load_default(size=24))
    for i, (im, text) in enumerate(zip([base, original, result], ['Edited base: enlarged', 'Registered original', 'Detail restored'])):
        x = 18 + i * (side + 18)
        draw.text((x, 61), text, fill='#111111', font=ImageFont.load_default(size=21))
        if im.mode == 'RGBA':
            backdrop = Image.new('RGB', im.size, '#bdbdbd')
            backdrop.paste(im, mask=im.getchannel('A'))
            im = backdrop
        board.paste(im.crop(box), (x, 104))
    draw.text((18, side + 120), 'Native-pixel comparison. Target appearance from edited base; no sharpening.',
              fill='#333333', font=ImageFont.load_default(size=16))
    board.save(out / (name + '-comparison.png'))


def fuse_legacy(args, regions):
    report = json.loads((args.out / 'report.json').read_text())
    for key, path in [('source', args.source), ('base', args.base), ('regions', args.regions)]:
        if fingerprint(path) != report[key + '_sha256']:
            raise ValueError(f'{key} changed after analysis; rerun --stage analyze')
    with np.load(args.out / 'registration.npz') as archive:
        data = {key: archive[key] for key in archive.files}
    source, edited = load_image(args.source), load_image(args.base)
    width, height = source.size
    base = edited.resize(source.size, Image.Resampling.LANCZOS)
    base_array = np.array(base)
    source_array = np.asarray(source)[..., :3]
    result = base_array.copy()
    registered = np.zeros((height, width, 3), np.uint8)
    outside_changed = 0
    low_sum, low_count = 0.0, 0
    # Frequency radii are proportional to source/working scale, not a fixed photo size.
    ratio = width / report['working_size'][0]
    fine_sigma, mid_sigma = max(.65, .54 * ratio), max(2, 3.2 * ratio)
    halo = int(np.ceil(5 * mid_sigma)) + 2
    for start in range(0, height, 512):
        end = min(height, start + 512)
        top, bottom = max(0, start - halo), min(height, end + halo)
        keep = slice(start - top, end - top)
        sx, sy, p = tile_data(data, width, height, top, bottom)
        warped = cv.remap(source_array, sx, sy, cv.INTER_CUBIC, borderMode=cv.BORDER_REFLECT_101)
        registered[start:end] = warped[keep]
        lb = cv.cvtColor(base_array[top:bottom, :, :3].astype(np.float32) / 255, cv.COLOR_RGB2Lab)
        ls = cv.cvtColor(warped.astype(np.float32) / 255, cv.COLOR_RGB2Lab)[..., 0]
        ls = blur(ls, .45)
        b2, b12 = blur(lb[..., 0], fine_sigma), blur(lb[..., 0], mid_sigma)
        s2, s12 = blur(ls, fine_sigma), blur(ls, mid_sigma)
        delta = p['fine'] * (p['gain'] * (ls - s2) - (lb[..., 0] - b2))
        delta += p['medium'] * (p['gain'] * (s2 - s12) - (b2 - b12))
        lb[..., 0] = np.clip(lb[..., 0] + delta, 0, 100)
        rgb = np.uint8(np.clip(np.rint(cv.cvtColor(lb, cv.COLOR_Lab2RGB) * 255), 0, 255))[keep]
        # No round-trip color conversion at all where both detail weights are zero.
        active = ((p['fine'][keep] > 0) | (p['medium'][keep] > 0)) & (base_array[start:end, :, 3] >= 250)
        result[start:end, :, :3][active] = rgb[active]
        protected = p['support'][keep] > 0
        outside_changed += int(np.any(result[start:end][~protected] != base_array[start:end][~protected], axis=1).sum())
        if protected.any():
            low_sum += float(np.abs(blur(delta, max(2, 10 * ratio))[keep][protected]).sum())
            low_count += int(protected.sum())
        log(f'Native-grid frequency fusion: {end}/{height} rows')
    if outside_changed:
        raise RuntimeError('Background preservation invariant failed')
    if not np.array_equal(result[..., 3], base_array[..., 3]):
        raise RuntimeError('Transparency preservation invariant failed')
    image = Image.fromarray(result)
    png = args.out / 'detail-restored.png'
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    image.save(png, icc_profile=profile)
    files = {png.name: png.stat().st_size}
    if np.all(result[..., 3] == 255):
        jpg = args.out / 'detail-restored.jpg'
        image.convert('RGB').save(jpg, quality=96, subsampling=0, optimize=True, icc_profile=profile)
        files[jpg.name] = jpg.stat().st_size
    preview = image.copy()
    preview.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
    preview.save(args.out / 'preview.png')
    for item in regions.get('comparisons', []):
        name, center = item['name'], item['center']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in name):
            raise ValueError('Comparison names must use lowercase ASCII letters, digits, hyphens')
        if len(center) != 2 or not all(0 <= float(c) <= 1 for c in center):
            raise ValueError('Comparison center must be normalized [x,y]')
        detail_crop(name, center, Image.fromarray(registered), base, image, args.out)
    with Image.open(png) as saved:
        saved.verify()
    report['validation'] = dict(outside_support_changed_pixels=outside_changed,
        alpha_preserved=True, png_verified=True, output_size=list(image.size),
        low_frequency_mean_absolute_delta_L=low_sum / max(1, low_count),
        note='Exact background/alpha preservation is relative to Lanczos-enlarged base in sRGB; JPEG is lossy.')
    report['files'] = files
    report['frequency_sigma_native_pixels'] = [fine_sigma, mid_sigma]
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    log('Saved: ' + json.dumps(files))


def fuse(args, regions):
    report=json.loads((args.out/'report.json').read_text())
    method=getattr(args,'method','local-rgb')
    if method!=report.get('restore_method','lab-detail'):
        raise ValueError('Restoration method changed after analysis; rerun --stage analyze')
    if method=='local-rgb':
        requested=getattr(args,'structure_channel','auto')
        if requested!=report.get('structure_channel_request','auto'):
            raise ValueError('Structure channel changed after analysis; rerun --stage analyze')
        import mapping_core
        mapping_core.fuse(args,regions,sys.modules[__name__])
    else:
        fuse_legacy(args,regions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--regions', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=['analyze', 'fuse', 'all'], default='all')
    parser.add_argument('--work-long-edge', type=int, default=1800)
    parser.add_argument('--method', choices=['local-rgb','lab-detail'], default='local-rgb',
                        help='Default: local RGB curve transfer; lab-detail is legacy explicit opt-in')
    parser.add_argument('--structure-channel', choices=['auto','L','R','G','B'], default='auto')
    args = parser.parse_args()
    if args.work_long_edge < 128:
        parser.error('--work-long-edge must be at least 128')
    args.out.mkdir(parents=True, exist_ok=True)
    protected_paths = {args.source.resolve(), args.base.resolve(), args.regions.resolve()}
    if any((args.out / name).resolve() in protected_paths for name in ['detail-restored.png', 'detail-restored.jpg']):
        parser.error('Output would overwrite an input; choose a separate job output directory')
    regions = json.loads(args.regions.read_text())
    cv.setNumThreads(4)
    cv.setRNGSeed(42)
    if args.stage in ['analyze', 'all']:
        analyze(args, regions)
    if args.stage in ['fuse', 'all']:
        fuse(args, regions)


if __name__ == '__main__':
    main()
