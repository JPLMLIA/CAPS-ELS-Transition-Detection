# Cassini CAPS ELS data reader
# Modeled after Gary's MDIS reader
# Kiri Wagstaff, 11/28/18

import os
from datetime import datetime
from collections import defaultdict
import numpy as np
from pds.core.parser import Parser
from scipy.interpolate import interp1d

GEOMFILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'ref',
    'geometricfactor.npz'
)
_EARRAY = None
_GEOM = None

E_CHARGE_COULOMBS = 1.602176487e-19
E_MASS_KG = 9.10938188e-31

def _load_gfactors():
    """
    Using global variables here because we only want to read these values from
    file once, then cache them at the module level
    """
    global _EARRAY
    global _GEOM
    if _EARRAY is None:
        sav = np.load(GEOMFILE)
        _EARRAY = sav['earray']
        _GEOM = sav['geom']

def needs_gfactors(f):
    """
    Decorator for any function that needs to have the geometric factors loaded
    first (calls `_load_gfactors` prior to calling the function).
    """
    def fprime(*args, **kwargs):
        _load_gfactors()
        return f(*args, **kwargs)
    return fprime

@needs_gfactors
def compute_def(e, counts):
    """
    Computes the Differential Energy Flux (DEF)
    Units: m^-2 sr^-1 s^-1

    According to Abi's script and the CAPS User Guide, this is done by dividing
    the counts by the anode- and energy-specific geometric factors.
    """

    # According to section 9.2 of the CAPS PDS User Guide, the proper thing to
    # do is interpolate the geometric factors: "If the ELS data record you are
    # working with has energy summing ... then you can use the above table to
    # interpolate the value you need for G."
    geom_interp = interp1d(
        _EARRAY, _GEOM, axis=0,
        fill_value='extrapolate',
        bounds_error=False,
        assume_sorted=True,
    )
    G = geom_interp(e)

    # newaxis is for the "phi" dimension of the data
    return counts / G[..., np.newaxis]

def compute_dnf(e, def_data):
    """
    Computes the Differential Number Flux (DNF)
    Units: m^-2 sr^-1 s^-1 J^-1

    Following Abi's script and the CAPS User Guide, this is the DEF divided by
    the product of the energy and the charge of the particle (electron).
    """
    # Add the new axes to broadcast across the theta/phi dimensions
    return def_data / (E_CHARGE_COULOMBS*e[..., np.newaxis, np.newaxis])

def compute_psd(e, def_data):
    """
    Computes the Phase Space Density (PSD)
    Units: m^-6 s^-3

    Following Abi's script and the CAPS User Guide, this is the DEF times a
    factor of (mass^2 / (2 q^2 E^2)).
    the product of the energy and the charge of the particle (electron).
    """
    qE_squared = (E_CHARGE_COULOMBS*e)**2
    # Add the new axes to broadcast across the theta/phi dimensions
    return (
        def_data * (E_MASS_KG**2) /
        (2 * qE_squared[..., np.newaxis, np.newaxis])
    )

def parse_dates(datearray):
    return np.array([
        datetime.strptime(row.tostring(), '%Y-%jT%H:%M:%S.%f')
        for row in datearray
    ])

def reshape_data(data):
    # Dimensions taken from ELS_V01.FMT
    # (records, energy, theta, phi)
    return data.reshape((-1, 63, 8, 1))

class ELS(object):

    COLUMNS = (
        # Values obtained from ELS_V01.FMT
        # Name, start byte, dtype, items, missing constant
        ('start_date',          1, np.uint8,    21,    None),
        ('dead_time_method',   22, np.uint8,     1,    None),
        ('record_dur',         25, np.float32,   1, 65535.0),
        ('acc_time',           29, np.float32,  63, 65535.0),
        ('data',              281, np.float32, 504, 65535.0),
        ('dim1_e',           2297, np.float32,  63, 65535.0),
        ('dim1_e_upper',     2549, np.float32,  63, 65535.0),
        ('dim1_e_lower',     2801, np.float32,  63, 65535.0),
        ('dim2_theta',       3053, np.float32,   8, 65535.0),
        ('dim2_theta_upper', 3085, np.float32,   8, 65535.0),
        ('dim2_theta_lower', 3117, np.float32,   8, 65535.0),
        ('dim3_phi',         3149, np.float32,   1, 65535.0),
        ('dim3_phi_upper',   3153, np.float32,   1, 65535.0),
        ('dim3_phi_lower',   3157, np.float32,   1, 65535.0),
    )

    POSTPROCESS = {
        'start_date': parse_dates,
        'data': reshape_data,
    }

    def __init__(self, data_path, lbl_path=None, verbose=False):
        """
        If the LBL file path is not specified, we'll assume that it is
        sitting right next to the DAT file (and raise an Error if not).
        """
        self.data_path = data_path
        if lbl_path is None:
            # Infer the LBL path if not supplied
            data_base, data_ext = os.path.splitext(data_path)
            if data_ext.lower() == data_ext:
                lbl_path = data_base + '.lbl'
            else:
                lbl_path = data_base + '.LBL'

        if not os.path.exists(lbl_path):
            raise ValueError('Expected LBL file "%s" does not exist' % lbl_path)

        self.lbl_path  = lbl_path
        self.verbose = verbose

        self._load()

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _load(self):
        with open(self.lbl_path, 'r') as f:
            parser = Parser()
            labels = parser.parse(f)

            record_bytes = int(labels['RECORD_BYTES'])
            nrecords     = int(labels['FILE_RECORDS'])

        columns = defaultdict(list)
        with open(self.data_path, 'rb') as f:
            for i in range(nrecords):
                for cname, cstart, ctype, citems, _ in ELS.COLUMNS:
                    # Subtract 1 because they are indexed from 1 in the .FMT
                    f.seek(i*record_bytes + cstart - 1)
                    columns[cname].append(f.read(np.dtype(ctype).itemsize*citems))

        for cname, _, ctype, citems, missing in ELS.COLUMNS:
            cstr = ''.join(columns[cname])
            col = np.fromstring(cstr, dtype=ctype, count=nrecords*citems)
            col = np.squeeze(col.reshape((nrecords, citems)))

            # Replace missing value with NaN
            if missing is not None:
                col[col == missing] = np.nan

            # Apply post-processing steps to appropriate columns
            if cname in ELS.POSTPROCESS:
                col = ELS.POSTPROCESS[cname](col)

            # Store column as object attribute
            setattr(self, cname, col)

        # Add iso_data by summing across theta/phi
        self.iso_data = np.sum(self.data, axis=(-2, -1))

        # Compute DEF, DNF, and PSD
        self.def_data = compute_def(self.dim1_e, self.data)
        self.dnf_data = compute_dnf(self.dim1_e, self.def_data)
        self.psd_data = compute_psd(self.dim1_e, self.def_data)
