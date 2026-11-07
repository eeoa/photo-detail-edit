"""Structure-gated local RGB curves learned at low resolution.

The native original supplies signal; the generated edit supplies the desired
local appearance. No Lab residual addition, no arbitrary unconstrained matrix.
Local-model/upsampling inspiration:
https://people.csail.mit.edu/kaiming/publications/pami12guidedfilter.pdf
The structural self-similarity descriptors are an approximation, not exact
invariance to arbitrary style changes or proof of semantic equivalence.
"""
import cv2 as cv
import numpy as np
import json
from PIL import Image, ImageCms
from cv_core import blur, smoothstep

def linear(a):
    a = np.asarray(a, np.float32) / 255
    return np.where(a <= .04045, a / 12.92, ((a + .055) / 1.055)**2.4)

def encoded(a):
    a = np.clip(a, 0, 1)
    a = np.where(a <= .0031308, 12.92 * a, 1.055 * a**(1/2.4) - .055)
    return np.uint8(np.clip(np.rint(a * 255), 0, 255))

def box(a, radius):
    return cv.boxFilter(a, -1, (2*radius+1, 2*radius+1), normalize=True,
                        borderType=cv.BORDER_REFLECT_101)

def self_similarity(log_intensity, sigma, offset):
    g = blur(log_intensity, sigma)
    d = []
    h, w = g.shape
    pad = np.pad(g, offset, mode='reflect')
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]:
        shifted = pad[offset+dy*offset:offset+dy*offset+h,
                      offset+dx*offset:offset+dx*offset+w]
        d.append(box((g-shifted)**2, 3))
    d = np.stack(d, axis=2)
    variance = np.maximum(d.mean(axis=2, keepdims=True), .002)
    desc = np.exp(-d / variance)
    gx, gy = cv.Sobel(g, cv.CV_32F, 1, 0, ksize=3), cv.Sobel(g, cv.CV_32F, 0, 1, ksize=3)
    return desc, gx, gy

def structural_match(source, target, channel):
    # Apply the same representation to both inputs. L means linear luminance;
    # a dominant monochromatic channel can be selected for saturated lighting.
    if channel=='L':
        weights=np.array([.2126,.7152,.0722],np.float32)
        a,b=source@weights,target@weights
    else:
        a,b=source[...,channel],target[...,channel]
    a,b=np.log(a+1e-4),np.log(b+1e-4)
    scores, errors = [], []
    for sigma, offset in [(2.0, 6), (4.0, 12)]:
        da, ax, ay = self_similarity(a, sigma, offset)
        db, bx, by = self_similarity(b, sigma, offset)
        error = blur(np.mean(np.abs(da-db), axis=2), 2)
        dot = np.abs(box(ax*bx + ay*by, 8))
        ea, eb = box(ax*ax + ay*ay, 8), box(bx*bx + by*by, 8)
        orientation = np.clip(dot / np.sqrt(ea*eb + 1e-6), 0, 1)
        # Smooth flat material has too little directional evidence; require
        # neighborhood self-similarity rather than a spurious edge direction.
        weak = np.minimum(ea, eb) < .002
        orientation[weak] = .75
        scores.append(.72*np.exp(-(error/.22)**2) + .28*orientation)
        errors.append(error)
    score = .55*scores[0] + .45*scores[1]
    weight = smoothstep(score, .43, .86)
    weight[np.maximum(errors[0], errors[1]) > .32] = 0
    return weight, score

def local_curves(source, target, weight, radius=24):
    """Positive monotonic per-channel power curves in overlapping neighborhoods.

    These are restricted local LUTs, not arbitrary color matrices. Positivity
    avoids negative RGB extrapolation and clipping holes at native edges.
    """
    x,y=np.log(np.maximum(source,1e-5)),np.log(np.maximum(target,1e-5))
    norm=np.maximum(box(weight,radius),1e-6)[...,None]
    w=weight[...,None]
    ex,ey=box(x*w,radius)/norm,box(y*w,radius)/norm
    variance=np.maximum(box(x*x*w,radius)/norm-ex*ex,0)
    covariance=box(x*y*w,radius)/norm-ex*ey
    ridge=.08+.15*variance
    gamma=np.clip((covariance+ridge)/(variance+ridge),.4,1.6)
    beta=ey-gamma*ex
    return gamma,beta

def apply_curves(source,gamma,beta):
    return np.exp(np.clip(beta+gamma*np.log(np.maximum(source,1e-6)),-16,3))


def choose_structure_channel(target, support, requested='auto'):
    if requested != 'auto':
        return requested
    pixels=target[support>.8]
    if not len(pixels):
        return 'L'
    energy=pixels.mean(axis=0)
    order=np.argsort(energy)
    return 'RGB'[order[-1]] if energy[order[-1]]>4*max(energy[order[-2]],1e-6) else 'L'


def fit(original_rgb, target_rgb, geometry, accepted, requested='auto'):
    """Learn six smooth curve parameters per neighborhood on a common grid."""
    h,w=geometry['support'].shape
    source=cv.resize(linear(original_rgb),(w,h),interpolation=cv.INTER_AREA)
    target=linear(np.asarray(Image.fromarray(target_rgb).resize((w,h),Image.Resampling.LANCZOS)))
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    ax,ay=xx+geometry['flow'][...,0],yy+geometry['flow'][...,1]
    inverse=cv.invertAffineTransform(geometry['matrix'])
    sx=inverse[0,0]*ax+inverse[0,1]*ay+inverse[0,2]
    sy=inverse[1,0]*ax+inverse[1,1]*ay+inverse[1,2]
    warped=np.clip(cv.remap(source,sx,sy,cv.INTER_CUBIC,borderMode=cv.BORDER_REFLECT_101),0,1)
    support=geometry['support']
    chosen=choose_structure_channel(target,support,requested)
    shape,score=structural_match(warped,target,chosen if chosen=='L' else 'RGB'.index(chosen))
    gate=np.minimum(blur(shape*support,1.5),support)
    gate[shape<=0]=0
    uy,ux=np.gradient(geometry['flow'][...,0]);vy,vx=np.gradient(geometry['flow'][...,1])
    gate*=np.exp(-(np.sqrt(ux*ux+uy*uy+vx*vx+vy*vy)/.7)**2)
    if not accepted:
        gate[:]=0
    training=gate.copy()
    source_blur,target_blur=blur(warped,1.5),blur(target,1.5)
    gamma,beta=local_curves(source_blur,target_blur,training)
    prediction=apply_curves(source_blur,gamma,beta)
    error=np.linalg.norm(prediction-target_blur,axis=2)
    robust=training*np.minimum(1,.025/(error+1e-5))
    gamma,beta=local_curves(source_blur,target_blur,robust)
    prediction=apply_curves(source_blur,gamma,beta)
    numerator=blur(target_blur*training[...,None],8)+1e-6
    denominator=blur(prediction*training[...,None],8)+1e-6
    beta+=blur(np.clip(np.log(numerator/denominator),-.7,.7),2)
    prediction=apply_curves(source_blur,gamma,beta)
    error=blur(np.linalg.norm(prediction-target_blur,axis=2),3)
    gate=np.clip(gate*np.exp(-(error/.08)**2),0,1)
    coefficients=blur(np.concatenate([gamma,beta],axis=2),1)
    size=(max(2,round(w/4)),max(2,round(h/4)))
    grid=np.concatenate([cv.resize(coefficients[...,i:i+3],size,interpolation=cv.INTER_AREA)
                          for i in [0,3]],axis=2)
    inside=support>.8
    fields=dict(coefficients=grid,gate=gate,shape_score=score,source_map_x=sx,source_map_y=sy)
    metadata=dict(restore_method='local-rgb',structure_channel_request=requested,
        structure_channel=chosen,mapping_model='positive_monotonic_spatial_RGB_power_curves',
        not_a_full_3d_lut=True,luminance_only=False,source_color_transfer=False,
        reliable_fraction=float((gate[inside]>.25).mean()) if inside.any() else 0,
        mean_mapping_weight=float(gate[inside].mean()) if inside.any() else 0,
        fit_error_mean=float(error[inside].mean()) if inside.any() else 0,
        transfer_enabled=bool(np.any(gate>0)),
        gamma_bounds=[.4,1.6],fit_window_radius_working_pixels=24)
    return fields,metadata


def fuse(args, regions, api):
    """Apply learned low-resolution curves to native source; preserve fallback."""
    report=json.loads((args.out/'report.json').read_text())
    for key,path in [('source',args.source),('base',args.base),('regions',args.regions)]:
        if api.fingerprint(path)!=report[key+'_sha256']:
            raise ValueError(f'{key} changed after analysis; rerun --stage analyze')
    field_file=args.out/'local-transform-field.npz'
    if api.fingerprint(field_file)!=report['mapping_sha256']:
        raise ValueError('Mapping fields changed after analysis; rerun --stage analyze')
    with np.load(args.out/'registration.npz')as z:
        support=z['support']
    with np.load(field_file)as z:
        fields={key:z[key]for key in z.files}
    source,edited=api.load_image(args.source),api.load_image(args.base)
    width,height=source.size
    h,w=support.shape
    base=edited.resize(source.size,Image.Resampling.LANCZOS)
    base_array=np.array(base)
    native=linear(np.asarray(source)[...,:3])
    result=base_array.copy()
    registered=np.zeros((height,width,3),np.uint8)
    grid=fields['coefficients'];gh,gw=grid.shape[:2]
    outside_changes=0
    for start in range(0,height,512):
        end=min(height,start+512)
        gx=(np.arange(width,dtype=np.float32)+.5)*w/width-.5
        gy=(np.arange(start,end,dtype=np.float32)+.5)*h/height-.5
        mx,my=np.meshgrid(gx,gy)
        def remap(a, interpolation=cv.INTER_LINEAR):
            return cv.remap(a,mx,my,interpolation,borderMode=cv.BORDER_REPLICATE)
        sx=(remap(fields['source_map_x'],cv.INTER_CUBIC)+.5)*width/w-.5
        sy=(remap(fields['source_map_y'],cv.INTER_CUBIC)+.5)*height/h-.5
        detail=cv.remap(native,sx,sy,cv.INTER_CUBIC,borderMode=cv.BORDER_REFLECT_101)
        registered[start:end]=encoded(detail)
        cx=(mx+.5)*gw/w-.5;cy=(my+.5)*gh/h-.5
        coef=np.concatenate([cv.remap(grid[...,i:i+3],cx,cy,cv.INTER_LINEAR,
            borderMode=cv.BORDER_REPLICATE)for i in [0,3]],axis=2)
        mapped=np.clip(apply_curves(np.maximum(detail,0),coef[...,:3],coef[...,3:]),0,1)
        weight=remap(fields['gate'])
        allowed=(remap(support)>0)&(base_array[start:end,:,3]>=250)
        weight[~allowed]=0
        active=weight>0
        fallback=linear(base_array[start:end,:,:3])
        rgb=encoded(weight[...,None]*mapped+(1-weight[...,None])*fallback)
        result[start:end,:,:3][active]=rgb[active]
        outside_changes+=int(np.any(result[start:end][~allowed]!=base_array[start:end][~allowed],axis=1).sum())
        api.log(f'Native-grid local RGB mapping: {end}/{height} rows')
    if outside_changes:
        raise RuntimeError('Background preservation invariant failed')
    if not np.array_equal(result[...,3],base_array[...,3]):
        raise RuntimeError('Transparency preservation invariant failed')
    excluded=api.polygon_mask((height,width),regions.get('base',{}).get('exclude_polygons',[]))>0
    excluded_changes=int(np.any(result[excluded]!=base_array[excluded],axis=1).sum())
    if excluded_changes:
        raise RuntimeError('Intentional edit exclusion invariant failed')
    image=Image.fromarray(result)
    profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    png=args.out/'detail-restored.png'
    image.save(png,icc_profile=profile)
    files={png.name:png.stat().st_size}
    if np.all(result[...,3]==255):
        jpg=args.out/'detail-restored.jpg'
        image.convert('RGB').save(jpg,quality=96,subsampling=0,optimize=True,icc_profile=profile)
        files[jpg.name]=jpg.stat().st_size
    preview=image.copy();preview.thumbnail((1800,1800),Image.Resampling.LANCZOS)
    preview.save(args.out/'preview.png')
    for item in regions.get('comparisons',[]):
        name,center=item['name'],item['center']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-'for c in name):
            raise ValueError('Comparison names must use lowercase ASCII letters, digits, hyphens')
        if len(center)!=2 or not all(0<=float(c)<=1 for c in center):
            raise ValueError('Comparison center must be normalized [x,y]')
        api.detail_crop(name,center,Image.fromarray(registered),base,image,args.out)
    with Image.open(png)as saved:
        saved.verify()
    report['validation']=dict(outside_support_changed_pixels=outside_changes,
        excluded_region_changed_pixels=excluded_changes,alpha_preserved=True,
        png_verified=True,output_size=list(image.size),
        note='Background, excluded areas and alpha match the Lanczos-enlarged sRGB base exactly in PNG. Subject color and lighting are fitted, not pixel-identical. JPEG is lossy.')
    report['files']=files
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    api.log('Saved: '+json.dumps(files))
