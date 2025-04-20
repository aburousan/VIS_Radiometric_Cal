import pvl
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from numba import njit, prange
import os
defective_vis_pixels = [
    (30, 308), (31, 308), (47, 409), (48, 187), (48, 188), (49, 59), (54, 137), (71, 215),
    (100, 78), (108, 413), (109, 19), (111, 19), (114, 424), (118, 363), (126, 410), (130, 292),
    (136, 271), (139, 235), (147, 222), (150, 54), (150, 59), (150, 78), (160, 372),
    (162, 36), (162, 37), (162, 248), (162, 330), (163, 36), (163, 37), (163, 248), (163, 330),
    (165, 32), (166, 32), (166, 173), (168, 232), (169, 363), (172, 189), (173, 92),
    (175, 228), (175, 266), (175, 267), (176, 152), (176, 229), (177, 155), (179, 196),
    (181, 249), (183, 354), (186, 238),
    (147, 222), (250, 223), (251, 223)  # Also flagged as filter boundaries + defective
]

defective_ir_pixels = [
    (149, 169), (149, 170), (155, 1), (156, 1), (156, 2), (156, 3), (156, 4), (156, 5),
    (156, 6), (156, 7), (156, 8), (156, 9), (156, 196),
    (157, 1), (157, 2), (157, 3), (157, 4), (157, 5), (157, 6), (157, 7), (157, 8),
    (157, 9), (157, 10), (157, 11), (157, 12), (157, 13), (157, 14), (157, 15),
    (157, 25),
    (158, 9), (158, 10), (158, 11), (158, 12), (158, 13), (158, 14), (158, 15), (158, 16), (158, 17),
    (159, 14), (159, 15), (159, 16), (159, 17), (159, 18),
    (160, 19), (160, 20), (160, 28), (160, 29),
    (161, 26), (161, 28), (161, 29), (161, 181),
    (171, 57), (171, 58), (171, 59), (171, 60), (171, 61), (171, 62), (171, 63), (171, 64),
    (172, 57), (172, 58), (172, 59), (172, 60), (172, 61), (172, 62), (172, 63), (172, 64),
    (172, 227),
    (173, 59), (173, 60), (173, 61), (173, 62), (173, 63), (173, 64), (173, 65), (173, 66), (173, 67), (173, 68),
    (174, 60), (174, 61), (174, 62), (174, 63), (174, 64), (174, 65), (174, 66), (174, 67),
    (175, 61), (175, 62), (175, 63),
    (191, 111), (191, 112),
    (192, 110), (192, 111), (192, 112), (192, 113),
    (193, 111), (193, 112), (193, 245), (193, 246),
    (219, 428),
    (227, 211),
    (228, 79), (228, 222),
    (229, 116),
    (234, 175), (235, 175), (235, 226),
    (236, 186),
    (237, 129),
    (238, 38),
    (241, 233),
    (243, 202),
    (244, 228),
    (245, 191), (245, 192),
    (250, 414)
]
def parse_lbl_metadata(lbl_path):
    lbl_data = pvl.load(lbl_path)
    qube = lbl_data["QUBE"]
    bands, samples, lines = qube["CORE_ITEMS"]
    item_bytes = qube["CORE_ITEM_BYTES"]
    item_type = qube["CORE_ITEM_TYPE"]
    dtype_map = {
        (2, "MSB_INTEGER"): ">i2",
        (2, "LSB_INTEGER"): "<i2",
        (4, "MSB_INTEGER"): ">i4",
        (4, "LSB_INTEGER"): "<i4",
        (4, "IEEE_REAL"): ">f4",
        (8, "IEEE_REAL"): ">f8"}
    dtype = dtype_map.get((item_bytes, item_type))
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {item_bytes} {item_type}")
    band_bin = qube.get("BAND_BIN", {})
    exposure_time = None
    if "FRAME_PARAMETER" in lbl_data:
        try:
            idx = lbl_data.get("FRAME_PARAMETER_DESC", []).index("EXPOSURE_DURATION")
            exposure_time = float(lbl_data["FRAME_PARAMETER"][idx])
        except (ValueError, IndexError):
            pass
    return {
        "shape": (bands, samples, lines),
        "dtype": dtype,
        "core_null": qube.get("CORE_NULL", -32768),
        "core_low_saturation": qube.get("CORE_LOW_REPR_SATURATION", -32767),
        "core_high_saturation": qube.get("CORE_HIGH_REPR_SATURATION", -32767),
        "core_multiplier": qube.get("CORE_MULTIPLIER", 1.0),
        "core_base": qube.get("CORE_BASE", 0.0),
        "product_type": lbl_data.get("PRODUCT_TYPE", "UNKNOWN"),
        "exposure_time": exposure_time,
        "wave_length_cen": np.array(band_bin.get("BAND_BIN_CENTER", []), dtype=np.float32),
        "wave_width": np.array(band_bin.get("BAND_BIN_WIDTH", []), dtype=np.float32),
        "wave_length_band_val": np.array(band_bin.get("BAND_BIN_ORIGINAL_BAND", []), dtype=np.int16),
        "spacecraft_solar_dist": qube.get("SPACECRAFT_SOLAR_DISTANCE", 441765159),}
def fix_endianness(array, dtype):
    dtype = np.dtype(dtype)
    if dtype.byteorder not in ('=', '|') and dtype.byteorder != sys.byteorder:
        array = array.byteswap().view(dtype.newbyteorder())
    return array
def load_qub_from_lbl(lbl_path, qub_path, cal=False):
    meta = parse_lbl_metadata(lbl_path)
    shape = meta["shape"]
    dtype = meta["dtype"]
    with open(qub_path, "rb") as f:
        data = np.fromfile(f, dtype=dtype)
    data = fix_endianness(data, dtype).reshape(shape, order="F")
    if cal:
        return data.astype(">f4"), meta
    data = data.astype(np.float32)
    null, low, high = meta["core_null"], meta["core_low_saturation"], meta["core_high_saturation"]
    mask = (data == null) | (data == low) | (data == high)
    data[mask] = np.nan
    mult, base = meta["core_multiplier"], meta["core_base"]
    if mult != 1.0 or base != 0.0:
        data *= mult
        data += base
    return data, meta
def extract_hk_data(lbl_hk_file, tab_hk_file):
    lbl_data = pvl.load(lbl_hk_file)
    columns = lbl_data["TABLE"].getlist("COLUMN")
    name_map = {
        "SHUTTER STATUS": "shutter_status",
        "IR TEMP": "ir_temp",
        "CCD TEMP": "ccd_temp",
        "SPECT TEMP": "spect_temp",
        "TELE TEMP": "tele_temp",
        "COLD TIP TEMP": "cold_tip_temp",
        "RADIATOR TEMP": "radiator_temp",
        "LEDGE TEMP": "ledge_temp",
        "CCD EXPO": "exposure_time_ccd",
        "IR EXPO": "exposure_time_ir",}
    column_specs = {}
    for col in columns:
        name = col["NAME"].strip().upper()
        if name in name_map:
            key = name_map[name]
            start = int(col["START_BYTE"]) - 1
            length = int(col["BYTES"])
            column_specs[key] = slice(start, start + length)
    missing = set(name_map.values()) - set(column_specs.keys())
    if missing:
        print(f"Warning: Missing columns in LBL: {missing}")
    extracted_data = {key: [] for key in column_specs}
    with open(tab_hk_file) as f:
        lines = f.readlines()
    shutter_open = {"open": 1, "closed": 0}
    for line in lines:
        for key, sl in column_specs.items():
            raw = line[sl].strip()
            if key == "shutter_status":
                extracted_data[key].append(shutter_open.get(raw.lower(), None))
            else:
                try:
                    extracted_data[key].append(float(raw))
                except ValueError:
                    extracted_data[key].append(None)
    shutter_data = extracted_data.get("shutter_status", [])
    closed_indexes = [i for i, val in enumerate(shutter_data) if val == 0]
    opened_indexes = [i for i, val in enumerate(shutter_data) if val == 1]
    return {
        "data": extracted_data,
        "closed_indexes": closed_indexes,
        "opened_indexes": opened_indexes,}
@njit(parallel=True)#line<->bands
def oversample_detilt_resample_cube(raw_cube):
    bands, samples, lines = raw_cube.shape
    raw_cube = raw_cube.astype(np.float32)
    raw_qube_detilt = np.zeros((bands, samples, lines), dtype=np.float32)
    for li in prange(lines):
        frame_expanded = np.zeros((bands, samples * 44), dtype=np.float32)
        frame_exp_detilt = np.zeros((bands, samples * 40), dtype=np.float32)
        for sa in range(samples):
            for ss in range(40):
                frame_expanded[:, sa * 40 + ss] = raw_cube[:, sa, li]
        for sa in range(samples):
            for ba in range(bands):
                bsh = ba // 4
                for ss in range(40):
                    idx_from = sa * 40 + ss + bsh
                    idx_to = sa * 40 + ss
                    if 0 <= idx_from < samples * 40:
                        frame_exp_detilt[ba, idx_to] = frame_expanded[ba, idx_from]
        for sa in range(samples):
            for ba in range(bands):
                total = 0.0
                for ss in range(40):
                    total += frame_exp_detilt[ba, sa * 40 + ss]
                raw_qube_detilt[ba, sa, li] = total / 40.0
    return raw_qube_detilt
def subtract_dark_frames(cube_array, closed_indexes, open_indexes):
    bands, samples, lines = cube_array.shape
    if not closed_indexes:
        raise ValueError("No dark (shutter=closed) frames found.")
    dark_frames = cube_array[:, :, closed_indexes]
    if dark_frames.shape[2] == 1:
        avg_dark = dark_frames[:, :, 0]
        dark_corrected_cube = cube_array[:, :, open_indexes] - avg_dark[:, :, None]
    else:
        interp_func = interp1d(
            closed_indexes,
            dark_frames,
            axis=2,
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate")
        interpolated_dark = interp_func(np.arange(lines))
        dark_corrected_cube = cube_array[:, :, open_indexes] - interpolated_dark[:, :, open_indexes]
    return dark_corrected_cube
def dn_to_radiance(dn_cube, itf, exposure_times):
    bands, samples, lines = dn_cube.shape
    assert exposure_times.shape[0] == lines
    radiance_cube = np.full_like(dn_cube, np.nan, dtype=np.float32)
    for l in range(lines):
        frame = dn_cube[:, :, l]
        expo = exposure_times[l]
        if np.isnan(expo) or expo <= 0:
            continue
        radiance_cube[:, :, l] = frame / (itf * expo)
    return radiance_cube
def to_zero_based(pixels):
    return [(s - 1, b - 1) for s, b in pixels]
defective_vis_pixels = to_zero_based(defective_vis_pixels)
#defective_ir_pixels = to_zero_based(defective_ir_pixels)
def mask_defective_pixels(cube, channel='VIS'):
    """
    Masks known defective pixels from the cube.

    Parameters:
    - cube: ndarray of shape (bands, samples, lines)
    - channel: 'VIS' or 'IR'
    """
    bands, samples, lines = cube.shape
    pixels = defective_vis_pixels if channel.upper() == 'VIS' else defective_ir_pixels
    for sample, band in pixels:
        if 0 <= band < bands and 0 <= sample < samples:
            cube[band, sample, :] = np.nan
    return cube
def load_ITF_data(filename, shape=(432, 256), dtype=np.float64):
    data = np.fromfile(filename, dtype=dtype).reshape(shape, order='F')
    return data.reshape(shape)
def create_band_to_wavelength_mapper(wvlen_band, wvlen_center):
    def linear_model(b, m, c):
        return m * b + c
    params, _ = curve_fit(linear_model, wvlen_band, wvlen_center)
    def band_to_wavelength_nm(band_index):
        return 10**3*linear_model(band_index, *params)
    return band_to_wavelength_nm
def compute_2_percent_limits(image):
    valid_pixels = image[np.isfinite(image)]
    low, high = np.percentile(valid_pixels, (2, 98))
    return low, high
#def plot_vir_transfer_function_with_wavelength(data, wv_b_map):
#    num_bands,num_samples = data.shape
#    wavelengths = np.array([wv_b_map(b) for b in range(num_bands)])
#    band_mask = (wavelengths > 231) & (wavelengths < 450)
#    masked_data = 10**3*data.copy()
#    masked_data[band_mask,:] = 0
#    clipped_data = masked_data[2:num_samples-1, :]
#    vmin, vmax = compute_2_percent_limits(clipped_data)
#    print(vmax,vmin)
#    plt.imshow(
#        masked_data.T,
#        aspect='auto',
#        cmap='gray',
#        origin='upper',
#        vmin=vmin,
#        vmax=vmax,
#        extent=[wavelengths[0], wavelengths[-1],1,num_samples-1])
#    plt.colorbar(label='Instrumental Transfer Function (DN m² nm sr / (W s))')
#    plt.xlabel('Wavelength [nm]')
#    plt.ylabel('Spatial')
#    plt.title('VIS ITF (masked: λ ∈ [231, 450] nm')
#    plt.show()
def load_vis_solar_irradiance(tab_path: str) -> np.ndarray:
    with open(tab_path, 'r') as f:
        irradiance = np.array([float(line.strip()) for line in f], dtype=">f4")

    if irradiance.shape[0] != 432:
        raise ValueError(f"Expected 432 bands, got {irradiance.shape[0]}")
    return irradiance
def compute_reflectance_from_calibrated_radiance(
    radiance_cube: np.ndarray,
    solar_irradiance: np.ndarray,
    spacecraft_solar_distance_km: float
) -> np.ndarray:
    K = 149597870.7  # Astronomical Unit in km
    scale_factor = (np.pi * (spacecraft_solar_distance_km ** 2)) / (K ** 2)
    si = solar_irradiance[:, np.newaxis, np.newaxis]
    reflectance = (radiance_cube * scale_factor) / si
    return reflectance
