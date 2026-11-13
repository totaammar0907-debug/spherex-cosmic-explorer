import io, base64
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits
from astroquery.ipac.irsa import Irsa
from astroquery.gaia import Gaia
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# بيانات احتياطية للأجرام الخافتة التي لا تظهر في Gaia
MANUAL = {
    "wise0855": dict(ra=133.786245, dec=-7.244372, parallax=438.9, pmra=-8118.0, pmdec=680.0)
}

def gaia_lookup(ra, dec, radius_deg=0.1):
    q = f"""
    SELECT TOP 5 source_id, ra, dec, parallax, pmra, pmdec,
           SQRT(pmra*pmra + pmdec*pmdec) AS pm_total
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
      AND parallax > 1
    ORDER BY pm_total DESC
    """
    r = Gaia.launch_job(q).get_results()
    if len(r) == 0:
        return None
    row = r
    return dict(ra=float(row['ra']), dec=float(row['dec']),
                parallax=float(row['parallax']),
                pmra=float(row['pmra']), pmdec=float(row['pmdec']))

def compute_trajectory(star, years_back=100, years_forward=100, step=10):
    c = SkyCoord(ra=star['ra']*u.deg, dec=star['dec']*u.deg,
                 distance=(1000/star['parallax'])*u.pc,
                 pm_ra_cosdec=star['pmra']*u.mas/u.yr,
                 pm_dec=star['pmdec']*u.mas/u.yr,
                 radial_velocity=0*u.km/u.s, obstime=Time('J2016.0'))
    path = []
    for dt in range(-years_back, years_forward + 1, step):
        n = c.apply_space_motion(new_obstime=Time(2016 + dt, format='jyear'))
        x, y, z = n.cartesian.xyz.to(u.pc).value
        path.append(dict(year=2016 + dt, ra=float(n.ra.deg), dec=float(n.dec.deg),
                         x=float(x), y=float(y), z=float(z), is_future=bool(dt > 0)))
    return dict(distance_pc=round(1000/star['parallax'], 2),
                distance_ly=round(3261.56/star['parallax'], 2), path=path)

def convert_to_base64_png(img_data, invert=False):
    vmin, vmax = np.nanpercentile(img_data,)
    if np.isnan(vmin) or np.isnan(vmax) or vmax <= vmin:
        norm_data = np.zeros_like(img_data, dtype=np.uint8)
    else:
        clipped = np.clip(img_data, vmin, vmax)
        clipped = np.nan_to_num(clipped, nan=vmin)
        norm_data = ((clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    if invert:
        norm_data = 255 - norm_data
    norm_data = np.flipud(norm_data)
    pil_img = Image.fromarray(norm_data, mode="L")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/api/star")
def get_star(ra: float, dec: float, name: str = ""):
    s = MANUAL.get(name.lower()) or gaia_lookup(ra, dec)
    if not s:
        raise HTTPException(status_code=404, detail="Star not found in Gaia")
    return {"star": s, **compute_trajectory(s)}

@app.get("/api/images")
def get_spherex_images(ra: float, dec: float, invert: bool = False):
    query = f"""
    SELECT 'https://caltech.edu' || a.uri || '?center={ra},{dec}d&size=0.05' AS uri,
           p.time_bounds_lower
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(POINT('ICRS', {ra}, {dec}), p.poly) AND p.energy_bandpassname = 'SPHEREx-D2'
    ORDER BY p.time_bounds_lower
    """
    results = Irsa.query_tap(query)
    if len(results) == 0:
        return {"images": []}
    indices = np.linspace(0, len(results) - 1, min(5, len(results)), dtype=int)
    images = []
    for idx in indices:
        with fits.open(results['uri'][idx]) as hdul:
            img_data = hdul.data
            images.append({
                "time": round(float(results['time_bounds_lower'][idx]), 2),
                "image": convert_to_base64_png(img_data, invert=invert)
            })
    return {"images": images}
