import streamlit as st
import numpy as np
import plotly.graph_objects as go
import tempfile
import os
# from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from utility_fn import (
    parse_lbl_metadata, load_qub_from_lbl, extract_hk_data,
    oversample_detilt_resample_cube, mask_defective_pixels,
    subtract_dark_frames, dn_to_radiance,
    load_ITF_data, create_band_to_wavelength_mapper,
    compute_reflectance_from_calibrated_radiance,
    load_vis_solar_irradiance
)
ABSORPTION_FEATURES = {
    'Titanium Dioxide (TiO2)': [420, 550, 900],  # Vesta, Ceres
    'Hematite (Fe2O3)': [500, 850],  # Vesta
    'Limonite (FeOOH)': [450, 850],  # Vesta
    'Magnetite (Fe3O4)': [420, 550],  # Vesta
    'Chromite (FeCr2O4)': [480, 650],  # Ceres, Vesta
    'Nickel Sulfides': [400, 650],  # Vesta
    'Sodium Sulfate (Mirabilite)': [500, 700],  # Ceres
    'Alunite (KAl3(SO4)2(OH)6)': [420, 650],  # Ceres
    'Iron Sulfide (Troilite)': [450, 650],  # Vesta
    'Diatomite (Hydrated Silica)': [400, 500],  # Ceres
    'Olivenite (Cu2(AsO4)(OH))': [600, 800],  # Vesta
    'Spinel (MgAl2O4)': [450, 550],  # Vesta
    'Andalusite (Al2SiO5)': [500, 700],  # Vesta
    'Diopside (CaMgSi2O6)': [450, 1000],  # Vesta
    'Feldspar (Alkali-rich varieties)': [550, 650],  # Ceres, Vesta
    'Albite (NaAlSi3O8)': [500, 650],  # Ceres
    'Sphalerite (ZnS)': [430, 530],  # Vesta
    'Forsterite (Mg2SiO4)': [420, 800],  # Vesta
    'Pyrargyrite (Ag3SbS3)': [460, 600],  # Vesta
    'Graphite (C)': [1500, 2700],  # Ceres, Vesta
    'Serpentine (Mg3(Si2O5)(OH)4)': [450, 700],  # Ceres
    'Sodium Chloride (NaCl)': [550, 650],  # Ceres
    'Sulfur Dioxide (SO2)': [430, 520],  # Ceres
    'Copper Oxide (CuO)': [600, 700],  # Vesta
    'Iron Chloride (FeCl2)': [430, 500],  # Ceres
    'Garnet (Almandine)': [450, 650],  # Vesta, Ceres
    'Kyanite (Al2SiO5)': [450, 700],  # Vesta
    'Molybdenite (MoS2)': [350, 600],  # Ceres, Vesta
    'Sodium Nitrate (NaNO3)': [430, 500],  # Ceres
    'Talc (Mg3Si4O10(OH)2)': [600, 800],  # Vesta
    'Monazite (CePO4)': [500, 600],  # Ceres, Vesta
    'Wollastonite (CaSiO3)': [400, 500],  # Vesta
    'Pyrite (FeS2)': [400, 500],  # Ceres, Vesta
    'Cobaltite (CoAsS)': [450, 600],  # Ceres, Vesta
    'Galena (PbS)': [350, 450],  # Ceres, Vesta
    'Cerussite (PbCO3)': [450, 600],  # Ceres, Vesta
    'Zircon (ZrSiO4)': [400, 500],  # Ceres
    'Rutile (TiO2)': [420, 550],  # Ceres, Vesta
    'Copper Sulfide (CuS)': [400, 600],  # Vesta
    'Sulfur (S)': [450, 520],  # Ceres, Vesta
    'Sodium Bicarbonate (NaHCO3)': [450, 500],  # Ceres
    'Sodium Carbonate (Na2CO3)': [2300, 2500],  # Ceres
    'Magnesium Sulfate (Epsomite)': [1400, 1900],  # Ceres
    'Magnesium Silicate (e.g., Serpentine)': [450, 700],  # Ceres
    'Calcium Sulfate (Gypsum)': [450, 600],  # Ceres
    'Vanadinite (Pb5(VO4)3Cl)': [450, 600],  # Ceres
    'Mica (e.g., Muscovite)': [400, 600],  # Vesta
    'Calcite (CaCO3)': [400, 600],  # Ceres, Vesta
    'Beryl (Be3Al2Si6O18)': [450, 550],  # Ceres
    'Wavellite (Al3(PO4)2(OH)5·H2O)': [400, 500],  # Ceres
    'Antimony Oxide (Sb2O3)': [400, 600],  # Vesta
    'Ferropericlase (Mg,Fe)O': [550, 600],  # Vesta
    'Chromium Oxide (Cr2O3)': [500, 600],  # Vesta
    'Staurolite (FeAl9O6(SiO4)4(OH)2)': [500, 600],  # Vesta
    'Titanite (CaTiSiO5)': [450, 600],  # Vesta
    'Orthopyroxene (Enstatite)': [450, 600],  # Ceres, Vesta
    'Labradorite (Plagioclase feldspar)': [450, 600],  # Ceres
    'Serpentine (Fe-Mg)': [600, 700],  # Ceres
    'Horneblende (Hydrated amphibole)': [600, 700],  # Ceres
    'Troilite (FeS)': [450, 500],  # Vesta
    'Chromium Sulfide (CrS)': [450, 500],  # Ceres
    'Nickel Oxide (NiO)': [450, 600],  # Vesta
    'Copper Oxide (CuO)': [600, 700],  # Vesta
    'Molybdite (MoO3)': [450, 600],  # Vesta
    'Calaverite (AuTe2)': [400, 500],  # Vesta
    'Tungsten Oxide (WO3)': [500, 600],  # Vesta
    'Graphene Oxide (C)': [450, 600],  # Ceres
    'Tungsten (W)': [450, 500],  # Vesta
    'Zinc Oxide (ZnO)': [350, 500],  # Ceres, Vesta
    'Lead Oxide (PbO)': [400, 500],  # Ceres
    'Magnesite (MgCO3)': [450, 600],  # Vesta
    'Cobalt Oxide (CoO)': [450, 600],  # Ceres, Vesta
    'Vanadium Oxide (V2O5)': [450, 600],  # Vesta
    'Phlogopite (Mica)': [400, 500],  # Ceres
    'Cinnabar (HgS)': [400, 500],  # Ceres
    'Ammonium Chloride (NH4Cl)': [450, 550],  # Ceres
    'Aluminium Oxide (Al2O3)': [500, 600],  # Ceres, Vesta
    'Sulfur Dioxide (SO2)': [430, 520],  # Ceres
    'Platinum (Pt)': [450, 500],  # Vesta
    'Iridium (Ir)': [400, 500],  # Vesta
    'Ruthenium (Ru)': [450, 550],  # Vesta
    'Osmium (Os)': [400, 500],  # Vesta
    'Manganese Oxide (MnO)': [450, 600],  # Vesta
    'Lithium Carbonate (Li2CO3)': [450, 500],  # Ceres
    'Potassium Nitrate (KNO3)': [450, 500],  # Ceres
    'Zirconium Oxide (ZrO2)': [500, 600],  # Vesta
    'Thorium Oxide (ThO2)': [450, 500],  # Ceres
    'Cerium Oxide (CeO2)': [500, 600],  # Ceres
    'Scandium Oxide (Sc2O3)': [450, 600],  # Ceres
    'Neodymium Oxide (Nd2O3)': [500, 600],  # Vesta
    'Yttrium Oxide (Y2O3)': [450, 600],  # Ceres
}
# def identify_absorption_features(wavelengths_nm, spectrum, prominence=0.005):
#     inv_spec = 1 - (spectrum / np.nanmax(spectrum))
#     peaks, _ = find_peaks(inv_spec, prominence=prominence)
#     found_features = []
#     for peak in peaks:
#         wl = wavelengths_nm[peak]
#         for material, features in ABSORPTION_FEATURES.items():
#             for ref_wl in features:
#                 if abs(wl - ref_wl) <= 20:
#                     found_features.append((wl, material))
#                     break
#     return found_features
st.set_page_config(layout="wide")
st.title("🌌 DAWN VIR Calibration Explorer")
st.markdown("**Author:** K.A.Rousan | JRF, NISER")

mode = st.sidebar.radio("Select Mode", ["VIS", "IR"])
lbl_file = st.file_uploader("Upload .LBL file", type="lbl")
qub_file = st.file_uploader("Upload .QUB file", type="qub")
lbl_hk_file = st.file_uploader("Upload Housekeeping .LBL", type="lbl")
tab_hk_file = st.file_uploader("Upload Housekeeping .TAB", type="tab")
itf_path_dict = {
    "VIS": "data/VIS/DAWN_VIR_VIS_RESP_V2.DAT",
    "IR": "data/IR/DAWN_VIR_IR_RESP_V2.DAT"
}
irrad_path_dict = {
    "VIS": "data/VIS/DAWN_VIR_VIS_SOLAR_SPECTRUM_V2.TAB",
    "IR": "data/IR/DAWN_VIR_VIS_SOLAR_SPECTRUM_V2.TAB"
}
def save_temp_file(uploaded_file, mode='wb'):
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode=mode) as tmp:
        if 'b' in mode:
            tmp.write(uploaded_file.read())
        else:
            tmp.write(uploaded_file.getvalue().decode('utf-8'))
        return tmp.name
@st.cache_resource
def full_pipeline(lbl_path, qub_path, lbl_hk_path, tab_hk_path, itf_path, solar_irradiance_path, mode):
    raw_cube, meta = load_qub_from_lbl(lbl_path, qub_path)
    band_to_wavelength = create_band_to_wavelength_mapper(meta['wave_length_band_val'], meta['wave_length_cen'])
    if mode == "VIS":
        detilted = oversample_detilt_resample_cube(raw_cube)
        masked = mask_defective_pixels(detilted, channel='VIS')
    else:
        masked = mask_defective_pixels(raw_cube, channel='IR')
    hk = extract_hk_data(lbl_hk_path, tab_hk_path)
    dark_corrected = subtract_dark_frames(masked, hk["closed_indexes"], hk["opened_indexes"])
    exposure_times = np.array(hk["data"]["exposure_time_ccd"])[hk["opened_indexes"]]
    itf = load_ITF_data(itf_path)
    radiance = dn_to_radiance(dark_corrected, itf, exposure_times)
    solar_irradiance = load_vis_solar_irradiance(solar_irradiance_path)
    spacecraft_solar_distance_km = meta["spacecraft_solar_dist"]
    reflectance = compute_reflectance_from_calibrated_radiance(radiance, solar_irradiance, spacecraft_solar_distance_km)
    result = {
        'meta': meta,
        'raw': raw_cube,
        'masked': masked,
        'dark': dark_corrected,
        'radiance': radiance,
        'reflectance': reflectance,
        'wavelength_map': band_to_wavelength
    }
    if mode == "VIS":
        result['detilted'] = detilted
    return result
if lbl_file and qub_file and lbl_hk_file and tab_hk_file:
    try:
        lbl_path = save_temp_file(lbl_file, mode='wb')
        qub_path = save_temp_file(qub_file, mode='wb')
        lbl_hk_path = save_temp_file(lbl_hk_file, mode='w')
        tab_hk_path = save_temp_file(tab_hk_file, mode='wb')
        itf_path = itf_path_dict[mode]
        solar_irradiance_path = irrad_path_dict[mode]
        st.success("Files saved, processing pipeline...")
        data = full_pipeline(lbl_path, qub_path, lbl_hk_path, tab_hk_path, itf_path, solar_irradiance_path, mode)
        bands, samples, lines = data["raw"].shape
        band_to_wavelength = data["wavelength_map"]
        st.sidebar.header("Display Settings")
        colormap_toggle = st.sidebar.checkbox("Enable Colormap Selection", value=True)
        colormaps = ['gray', 'viridis', 'hot', 'jet', 'inferno']
        cmap = st.sidebar.selectbox("Select Colormap", options=colormaps, index=0) if colormap_toggle else "gray"
        crop_enabled = st.sidebar.checkbox("Enable Image Cropping", value=False)
        crop_x_min = st.sidebar.number_input("Crop X Start", 0, samples-1, 0) if crop_enabled else 0
        crop_x_max = st.sidebar.number_input("Crop X End", crop_x_min+1, samples, samples) if crop_enabled else samples
        crop_y_min = st.sidebar.number_input("Crop Y Start", 0, lines-1, 0) if crop_enabled else 0
        crop_y_max = st.sidebar.number_input("Crop Y End", crop_y_min+1, lines, lines) if crop_enabled else lines
        st.sidebar.header("Spectral Profile Location")
        selected_sample = st.sidebar.slider("Select Sample (X)", 0, samples - 1, samples // 2)
        selected_line = st.sidebar.slider("Select Line (Y)", 0, lines - 1, lines // 2)
        cube_dict = {
            "Raw": data["raw"],
            "Masked": data["masked"],
            "Dark Subtracted": data["dark"],
            "Radiometrically Calibrated": data["radiance"]
        }
        if mode == "VIS":
            cube_dict["Detilted"] = data["detilted"]
        col_select1, col_select2 = st.columns(2)
        with col_select1:
            left_image = st.selectbox("Select Left Image", list(cube_dict.keys()), index=0)
        with col_select2:
            right_image = st.selectbox("Select Right Image", list(cube_dict.keys())[1:], index=0)
        band_slider = st.slider("Select Band Number", 0, bands - 1, bands // 2)
        col1, col2 = st.columns(2)
        for col, label in zip([col1, col2], [left_image, right_image]):
            with col:
                st.subheader(f"{label} Image")
                image = cube_dict[label][band_slider, crop_x_min:crop_x_max, crop_y_min:crop_y_max]
                vmin, vmax = np.nanpercentile(image, [2, 98])
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(image.T, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
                ax.set_title(f"{label} - Band {band_slider} ({band_to_wavelength(band_slider)*1e3:.1f} nm)")
                ax.set_xlabel("Sample")
                ax.set_ylabel("Line")
                st.pyplot(fig)
        st.sidebar.header("Plotly Charts")
        plotly_options = st.sidebar.multiselect(
            "Select Plotly Graphs to Display",
            ["Spectral Radiance", "Reflectance"],
            default=["Spectral Radiance", "Reflectance"]
        )
        wavelengths = np.array([band_to_wavelength(b) for b in range(bands)])
        if "Spectral Radiance" in plotly_options:
            st.subheader("Spectral Profile vs Wavelength")
            spectral_v = data["radiance"][:, selected_sample, selected_line]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wavelengths, y=spectral_v,
                                     mode='lines+markers', name='Radiance',
                                     line=dict(color='blue')))
            fig.update_layout(title=f"Spectral Profile at Pixel ({selected_sample}, {selected_line})",
                              xaxis_title="Wavelength (nm)",
                              yaxis_title="Spectral Radiance (W m⁻² μm⁻¹ sr⁻¹)",
                              xaxis_showgrid=True, yaxis_showgrid=True)
            st.plotly_chart(fig, use_container_width=True)
            # st.subheader("Identified Absorption Features (from Radiance)")
            # features = identify_absorption_features(wavelengths * 1000, spectral_v)
            # if features:
            #     for wl, material in features:
            #         st.markdown(f"- **{material}** near **{wl:.1f} nm**")
            # else:
            #     st.write("No known features matched.")
        if "Reflectance" in plotly_options:
            st.subheader("Reflectance vs Wavelength")
            reflectance_data = data["reflectance"][:, selected_sample, selected_line]
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=wavelengths, y=reflectance_data,
                                      mode='lines+markers', name='Reflectance',
                                      line=dict(color='green')))
            fig1.update_layout(title=f"Reflectance at Pixel ({selected_sample}, {selected_line})",
                               xaxis_title="Wavelength (nm)",
                               yaxis_title="Reflectance",
                               yaxis=dict(tickformat=".3e"),
                               hovermode="x unified",
                               xaxis_showgrid=True, yaxis_showgrid=True)
            st.plotly_chart(fig1, use_container_width=True)
        qub_filename = os.path.splitext(os.path.basename(qub_file.name))[0] + "_1B.npy"
        radiance_data = data["radiance"]
        buf = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
        np.save(buf.name, radiance_data)
        with open(buf.name, "rb") as f:
            st.download_button("Download Radiance Cube (.npy)", data=f, file_name=qub_filename, mime="application/octet-stream")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload required files to proceed with processing.")
