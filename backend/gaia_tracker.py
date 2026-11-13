import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.gaia import Gaia

# أهداف ما بتظهرش في Gaia: قيم يدوية (mas, mas/yr)
MANUAL = {
    "wise0855": dict(ra=133.786245, dec=-7.244372, parallax=440.8,
                     pmra=-8100.0, pmdec=700.0)  # راجع القيم!
}

def gaia_lookup(ra, dec, radius_deg=0.1):
    q = f"""
    SELECT TOP 5 source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(POINT('ICRS', ra, dec),
                     CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
      AND parallax > 1
    ORDER BY ABS(pmra) + ABS(pmdec) DESC
    """
    job = Gaia.launch_job(q)
    r = job.get_results()
    if len(r) == 0:
        return None
    row = r[0]
    return dict(ra=float(row['ra']), dec=float(row['dec']),
                parallax=float(row['parallax']),
                pmra=float(row['pmra']), pmdec=float(row['pmdec']))

def compute_trajectory(star, years_back=100, years_forward=100, step=10):
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
        path.append(dict(year=2016+dt, ra=n.ra.deg, dec=n.dec.deg,
                         x=x, y=y, z=z, is_future=dt > 0))
    return dict(distance_pc=1000/star['parallax'],
                distance_ly=3261.56/star['parallax'], path=path) 
