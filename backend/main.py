import io, base64
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits
from astroquery.ipac.irsa import Irsa
from astroquery.gaia import Gaia
import matplotlib
matplotlib.use('Agg') # ضروري لعمل الرسم على السيرفر بدون شاشة
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# بيانات احتياطية للأجرام الخافتة
MANUAL = {
    "wise0855": dict(ra=133.786245, dec=-7.244372, parallax=438.9, pmra=-8118.0, pmdec=680.0)
}

def gaia_lookup(ra, dec, radius_deg=0.1):
    q = f"""
    SELECT TOP 1 ra, dec, parallax, pmra, pmdec, (ABS(pmra) + ABS(pmdec)) AS total_pm
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
      AND parallax > 1
    ORDER BY total_pm DESC
    """
    r = Gaia.launch_job(q).get_results()
    if len(r) == 0:
        return None
    row = r[0]
    return dict(ra=float(row['ra']), dec=float(row['dec']),
                parallax=float(row['parallax']),
                pmra=float(row['pmra']), pmdec=float(row['pmdec']))

def compute_trajectory(star, years_back=100, years_forward=100, step=20):
    c = SkyCoord(ra=star['ra']*u.deg, dec=star['dec']*u.deg,
                 distance=(1000/star['parallax'])*u.pc,
                 pm_ra_cosdec=star['pmra']*u.mas/u.yr,
                 pm_dec=star['pmdec']*u.mas/u.yr,
                 radial_velocity=0*u.km/u.s,
                 obstime=Time('J2016.0'))
    path = []
    for dt in range(-years_back, years_forward + 1, step):
        n = c.apply_space_motion(new_obstime=Time(2016 + dt, format='jyear'))
        x, y, z = n.cartesian.xyz.to(u.pc).value
        path.append(dict(year=2016+dt, ra=float(n.ra.deg), dec=float(n.dec.deg),
                         x=float(x), y=float(y), z=float(z), is_future=bool(dt > 0)))
    return dict(distance_pc=round(1000/star['parallax'], 2),
                distance_ly=round(3261.56/star['parallax'], 2), path=path)

@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/api/star")
def get_star(ra: float, dec: float, name: str = ""):
    s = MANUAL.get(name.lower()) or gaia_lookup(ra, dec)
    if not s:
        raise HTTPException(404, "Star not found in Gaia")
    return {"star": s, **compute_trajectory(s)}

@app.get("/api/images")
def get_spherex_images(ra: float, dec: float, invert: bool = False):
    query = f"""
    SELECT 'https://irsa.ipac.caltech.edu/' || a.uri || '?center={ra},{dec}d&size=0.05' AS uri,
           p.time_bounds_lower
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(POINT('ICRS', {ra}, {dec}), p.poly)
          AND p.energy_bandpassname = 'SPHEREx-D2'
    ORDER BY p.time_bounds_lower
    """
    results = Irsa.query_tap(query)
    if len(results) == 0:
        return {"images": []}

    indices = np.linspace(0, len(results) - 1, min(5, len(results)), dtype=int)
    images = []
    cmap = 'Greys' if invert else 'inferno'

    for idx in indices:
        with fits.open(results['uri'][idx]) as hdul:
            img_data = hdul[1].data
            vmin, vmax = np.nanpercentile(img_data, [5, 99])
            fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
            ax.imshow(img_data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            images.append({
                "time": round(float(results['time_bounds_lower'][idx]), 2),
                "image": f"data:image/png;base64,{b64}"
            })
    return {"images": images}
