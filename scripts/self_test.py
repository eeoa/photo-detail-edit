#!/usr/bin/env python3
"""Offline behavioral checks using synthetic pixels, with no model/API costs."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2 as cv
import numpy as np
from PIL import Image

import restore_detail as rd


def run_case(root, name, original, edited, mask, exclusions=None):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(original).save(folder / 'source.png')
    Image.fromarray(edited).save(folder / 'base.png')
    Image.fromarray(mask * 255).save(folder / 'mask.png')
    config = {'source': {'mask': 'mask.png'},
              'base': {'mask': 'mask.png', 'exclude_polygons': exclusions or []}}
    (folder / 'regions.json').write_text(json.dumps(config))
    args = SimpleNamespace(source=folder / 'source.png', base=folder / 'base.png',
        regions=folder / 'regions.json', out=folder / 'out', work_long_edge=600)
    args.out.mkdir(exist_ok=True)
    rd.analyze(args, config)
    rd.fuse(args, config)
    report = json.loads((args.out / 'report.json').read_text())
    output = np.array(Image.open(args.out / 'detail-restored.png').convert('RGBA'))
    expected = np.array(rd.load_image(args.base).resize(
        Image.open(args.source).size, Image.Resampling.LANCZOS))
    assert output.shape == expected.shape
    assert np.array_equal(output[..., 3], expected[..., 3]), name + ': alpha changed'
    assert report['validation']['outside_support_changed_pixels'] == 0
    return args, report, output, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cv.setNumThreads(2)
    cv.setRNGSeed(42)
    rng = np.random.default_rng(815)
    h, w = 960, 640
    y, x = np.mgrid[:h, :w]
    mask = ((x > 110) & (x < 510) & (y > 120) & (y < 850)).astype(np.uint8)
    # Printed coarse marks enable feature matching; native fine grain is lost by resizing.
    coarse = rng.uniform(25, 220, (120, 80, 3)).astype(np.float32)
    photo = cv.resize(coarse, (w, h), interpolation=cv.INTER_CUBIC)
    photo += rng.normal(0, 7, (h, w, 1))
    photo = np.uint8(np.clip(photo, 0, 255))
    photo[mask == 0] = [202, 193, 182]
    small = np.array(Image.fromarray(photo).resize((320, 480), Image.Resampling.LANCZOS))
    blue = np.uint8(np.clip(small.astype(float) * [.70, .85, 1.08] + [2, 3, 10], 0, 255))
    # Intentional changed region must be left exactly as edited, including round trips.
    excluded = [[[.35, .35], [.60, .35], [.60, .60], [.35, .60]]]
    first, report, result, expected = run_case(args.out, 'opaque-blue', photo, blue, mask, excluded)
    assert report['transfer_enabled'], 'Positive textured correspondence was rejected'
    roi = (slice(round(.37 * h), round(.58 * h)), slice(round(.37 * w), round(.58 * w)))
    assert np.array_equal(result[roi], expected[roi]), 'Intentional edit exclusion changed'
    assert np.count_nonzero(result != expected) > 1000, 'No measured detail was transferred'
    assert report['restore_method']=='local-rgb', 'Default method did not switch'
    assert report['structure_channel']=='L', 'Multicolor blue photo should use luminance structure'
    # Native RGB detail legitimately differs from an enlarged low-resolution
    # base. Compare broad color/lighting to the known high-resolution edit, and
    # check that genuine fine-detail error falls rather than scoring pixelwise
    # chroma identity to a blurred reference.
    lout = cv.cvtColor(result[..., :3].astype(np.float32) / 255, cv.COLOR_RGB2Lab)
    lbase = cv.cvtColor(expected[..., :3].astype(np.float32) / 255, cv.COLOR_RGB2Lab)
    truth=np.uint8(np.clip(photo.astype(float)*[.70,.85,1.08]+[2,3,10],0,255))
    ltruth=cv.cvtColor(truth.astype(np.float32)/255,cv.COLOR_RGB2Lab)
    coarse_error=cv.GaussianBlur(lout-ltruth,(0,0),12)
    assert np.mean(np.abs(coarse_error[...,0]))<.3, 'Broad edited lighting was lost'
    assert np.mean(np.abs(coarse_error[...,1:]))<.3, 'Broad edited color cast was lost'
    high=lambda a:a-cv.GaussianBlur(a,(0,0),2)
    assert np.mean((high(lout)-high(ltruth))**2)<.9*np.mean((high(lbase)-high(ltruth))**2), 'Genuine fine-detail error did not improve'
    rgba = np.dstack([blue, cv.resize(mask * 255, (320, 480), interpolation=cv.INTER_LINEAR)])
    _, tr, _, _ = run_case(args.out, 'transparent', photo, rgba, mask)
    assert tr['validation']['alpha_preserved']
    assert 'detail-restored.jpg' not in tr['files'], 'Transparent photo was flattened to JPEG'
    neutral=np.repeat(photo[...,:1],3,axis=2)
    neutral_small=np.array(Image.fromarray(neutral).resize((320,480),Image.Resampling.LANCZOS))
    red=neutral_small.copy()
    red[...,1:]=0
    _,rr,red_result,_=run_case(args.out,'pure-red',neutral,red,mask)
    assert rr['transfer_enabled'] and rr['structure_channel']=='R'
    assert red_result[...,1:3].max()==0, 'Pure-red illumination gained gray/green/blue speckles'
    # Unrelated source content must not be forced into a similarly sized object.
    unrelated = rng.integers(0, 256, blue.shape, dtype=np.uint8)
    _, bad, unchanged, unchanged_base = run_case(args.out, 'unrelated', photo, unrelated, mask)
    assert not bad['transfer_enabled'], 'Unrelated input unexpectedly accepted'
    assert np.array_equal(unchanged, unchanged_base), 'Rejected input modified edited pixels'
    # A changed region file cannot replay stale registration.
    first.regions.write_text(json.dumps({'source': {}}))
    try:
        rd.fuse(first, {})
    except ValueError as error:
        assert 'changed after analysis' in str(error)
    else:
        raise AssertionError('Stale analysis replay was accepted')
    print('PASS: default local RGB mapping, true fine-detail error reduction, broad color/lighting retention, edit exclusion, transparency, pure-red retention, unrelated-input rejection, stale-analysis rejection')


if __name__ == '__main__':
    main()
