# functions/plotter_master.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.colors import to_hex
from shapely.geometry import Point

# =============================================================================
# 1. SETUP DAN PEMUATAN DATA UTAMA (DI-CACHE)
# =============================================================================

# FUNGSI BARU YANG SUDAH DIPERBAIKI
@st.cache_data(show_spinner=False)
def load_geospatial_data():
    prov_gdf = gpd.read_file("shapefile/OSM/Batas_Provinsi_Laut_2024_OSM_LapakGIS.shp")
    iho_gdf = gpd.read_file("shapefile/IHO/World_Seas_IHO_v3.shp").to_crs("EPSG:4326")

    # PERBAIKAN: Terapkan buffer hanya pada kolom 'geometry'
    # Ini memastikan 'prov_gdf' tetap menjadi GeoDataFrame
    original_crs = prov_gdf.crs
    prov_gdf['geometry'] = prov_gdf.to_crs(epsg=3857).geometry.buffer(100000).to_crs(original_crs)

    return prov_gdf, iho_gdf

@st.cache_data(show_spinner=False)
def load_timeseries_data(data_type: str):
    """
    Memuat data timeseries dari file Parquet berdasarkan tipe (observasi/proyeksi).
    Fungsi ini di-cache, sehingga setiap file hanya dibaca dari disk sekali.
    """
    if data_type == 'observasi':
        path = "hasil_tml_obs_by_wilayah.parquet"
    elif data_type == 'proyeksi':
        path = "data/canesm5_tml_proj245_by_wilayah.parquet"
    else:
        raise ValueError("Tipe data tidak valid. Pilih 'observasi' atau 'proyeksi'.")
        
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year

    # Cast kolom teks penting
    for col in ["provinsi", "kabupaten", "kecamatan", "desa"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    return df

# Memuat data geospasial sekali di awal
PROVINSI_GDF, IHO_GDF = load_geospatial_data()

# =============================================================================
# 2. FUNGSI-FUNGSI HELPER INTERNAL
# =============================================================================

def _get_region(gdf, lat, lon, region_col_names, default_val):
    """Fungsi generik untuk mendapatkan wilayah dari koordinat."""
    point = Point(lon, lat)
    point_gdf = gpd.GeoDataFrame(index=[0], geometry=[point], crs="EPSG:4326")
    match = gpd.sjoin(point_gdf, gdf, how="left", predicate="within")
    
    if match.empty or match.iloc[0].isnull().any():
        return default_val
        
    for col in region_col_names:
        if col in match.columns and pd.notna(match.iloc[0][col]):
            return match.iloc[0][col]
            
    return default_val

# Dictionary terjemahan nama laut
SEA_NAME_TRANSLATION = {
    "Arafura Sea": "Laut Arafura", "Banda Sea": "Laut Banda", "Java Sea": "Laut Jawa",
    "Savu Sea": "Laut Sawu", "Bali Sea": "Laut Bali", "Flores Sea": "Laut Flores",
    "Ceram Sea": "Laut Seram", "Bismarck Sea": "Laut Bismarck", "Gulf of Tomini": "Teluk Tomini",
    "Makassar Strait": "Selat Makassar", "Halmahera Sea": "Laut Halmahera", "Molukka Sea": "Laut Maluku",
    "Celebes Sea": "Laut Sulawesi", "Sulu Sea": "Laut Sulu", "Gulf of Thailand": "Teluk Thailand",
    "South China Sea": "Laut Cina Selatan", "Singapore Strait": "Selat Singapura",
    "Malacca Strait": "Selat Malaka", "Andaman or Burma Sea": "Laut Andaman",
    "Philippine Sea": "Laut Filipina", "Timor Sea": "Laut Timor", "Indian Ocean": "Samudra Hindia",
    "Pacific Ocean": "Samudra Pasifik"
}

def _translate_sea_name(name):
    return SEA_NAME_TRANSLATION.get(name, name)

def _extract_top_n(text, default=10):
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else default

# =============================================================================
# 3. FUNGSI PLOTTING PUBLIK (Dipanggil oleh Chatbot)
# =============================================================================

### ----- FUNGSI PLOT TIME SERIES & TREN ----- ###

def _generic_timeseries_plot(data_type, level, name, tahun=None, return_df=False):
    """Fungsi internal generik untuk plot time series per lokasi."""
    df = load_timeseries_data(data_type)
    df_filtered = df[df[level].str.lower() == name.lower()]
    if tahun:
        df_filtered = df_filtered[df_filtered["year"] == int(tahun)]
    if df_filtered.empty: return None, None
    
    df_grouped = df_filtered.groupby("time")["sla"].mean().reset_index()
    data_type_title = "Proyeksi " if data_type == 'proyeksi' else ""
    title = f"{data_type_title}TML {level.title()} {name.title()}" + (f" Tahun {tahun}" if tahun else "")
    fig = px.line(df_grouped, x="time", y="sla", title=title, labels={"sla": "Tinggi Muka Laut (m)", "time": "Waktu"})
    return (fig, df_grouped) if return_df else fig

def _generic_trend_plot(data_type, level, name, return_df=False):
    """Fungsi internal generik untuk membuat plot tren."""
    df_full = load_timeseries_data(data_type)
    df_filtered = df_full[df_full[level].str.lower() == name.lower()]
    if df_filtered.empty: return (None, None, None) if return_df else None
        
    df_yearly = df_filtered.groupby("year")["sla"].mean().reset_index()
    slope = np.polyfit(df_yearly["year"], df_yearly["sla"], 1)[0] * 1000 # konversi ke mm/tahun
    
    data_type_title = "Proyeksi " if data_type == 'proyeksi' else ""
    title = f"Tren {data_type_title}TML {level.title()} {name.title()}"
    fig = px.line(df_yearly, x="year", y="sla", title=title, markers=True, labels={"sla": "Tinggi Muka Laut (m)", "year": "Tahun"})
    
    return (fig, df_yearly, slope) if return_df else fig

def plot_tml_desa(desa, tahun=None, return_df=False):
    return _generic_timeseries_plot('observasi', 'desa', desa, tahun, return_df)

def plot_proyeksi_tml_desa(desa, tahun=None, return_df=False):
    return _generic_timeseries_plot('proyeksi', 'desa', desa, tahun, return_df)

def tren_tml_desa(desa, return_df=False):
    return _generic_trend_plot('observasi', 'desa', desa, return_df)

def tren_proyeksi_tml_desa(desa, return_df=False):
    return _generic_trend_plot('proyeksi', 'desa', desa, return_df)

def tren_tml_kecamatan(kecamatan, return_df=False):
    return _generic_trend_plot('observasi', 'kecamatan', kecamatan, return_df)

def tren_proyeksi_tml_kecamatan(kecamatan, return_df=False):
    return _generic_trend_plot('proyeksi', 'kecamatan', kecamatan, return_df)

def tren_tml_kabupaten(kabupaten, return_df=False):
    return _generic_trend_plot('observasi', 'kabupaten', kabupaten, return_df)

def tren_proyeksi_tml_kabupaten(kabupaten, return_df=False):
    return _generic_trend_plot('proyeksi', 'kabupaten', kabupaten, return_df)

### ----- FUNGSI PLOT BAR CHART ----- ###

def _create_yearly_timeseries_chart(data_type, level, name, tahun, return_df=False):
    """
    Fungsi internal generik untuk membuat LINE CHART rata-rata bulanan
    untuk satu wilayah (kabupaten/kecamatan) pada tahun tertentu.
    """
    df_full = load_timeseries_data(data_type)
    
    df_filtered = df_full[
        (df_full[level] == name.lower()) &
        (df_full["year"] == int(tahun))
    ]
    
    if df_filtered.empty:
        return None, None
        
    # Agregasi: Hitung rata-rata TML untuk seluruh wilayah per tanggal (bulanan)
    df_avg = df_filtered.groupby("time")["sla"].mean().reset_index()
    
    title_prefix = "Proyeksi " if data_type == 'proyeksi' else ""
    title = f"Rata-rata Bulanan {title_prefix}TML di {level.title()} {name.title()} ({tahun})"
    fig = px.line(df_avg, x="time", y="sla", title=title, labels={"sla": "TML Rata-rata (m)", "time": "Bulan"}, markers=True)
    
    return (fig, df_avg) if return_df else fig

def grafik_tahunan_kabupaten(kabupaten, tahun, return_df=False):
    return _create_yearly_timeseries_chart('observasi', 'kabupaten', kabupaten, tahun, return_df)

def grafik_proyeksi_tahunan_kabupaten(kabupaten, tahun, return_df=False):
    return _create_yearly_timeseries_chart('proyeksi', 'kabupaten', kabupaten, tahun, return_df)

def grafik_tahunan_kecamatan(kecamatan, tahun, return_df=False):
    return _create_yearly_timeseries_chart('observasi', 'kecamatan', kecamatan, tahun, return_df)

def grafik_proyeksi_tahunan_kecamatan(kecamatan, tahun, return_df=False):
    return _create_yearly_timeseries_chart('proyeksi', 'kecamatan', kecamatan, tahun, return_df)

### ----- FUNGSI PLOT TREN NASIONAL ----- ###

def _generic_national_trend_plot(data_type, return_df=False):
    """Fungsi internal generik untuk membuat plot tren nasional."""
    df_full = load_timeseries_data(data_type)
    if df_full.empty: return (None, None, None) if return_df else None
    
    df_yearly = df_full.groupby("year")["sla"].mean().reset_index()
    slope = np.polyfit(df_yearly["year"], df_yearly["sla"], 1)[0] * 1000 # konversi ke mm/tahun
    
    data_type_title = "Proyeksi " if data_type == 'proyeksi' else ""
    period = "2025-2100" if data_type == 'proyeksi' else "1993-2023"
    title = f"Tren {data_type_title}TML Rata-Rata Nasional ({period})"
    
    fig = px.line(df_yearly, x="year", y="sla", title=title, markers=True, labels={"sla": "Tinggi Muka Laut (m)", "year": "Tahun"})
    
    return (fig, df_yearly, slope) if return_df else fig

def tren_tml_nasional(return_df=False):
    return _generic_national_trend_plot('observasi', return_df)

def tren_proyeksi_tml_nasional(return_df=False):
    return _generic_national_trend_plot('proyeksi', return_df)

### ----- FUNGSI PLOT TAHUNAN & PERBANDINGAN ----- ###

def plot_tml_tahunan(tahun, return_df=False):
    df = load_timeseries_data('observasi')
    df_tahun = df[df["year"] == int(tahun)]
    if df_tahun.empty: return (None, None) if return_df else None
    df_grouped = df_tahun.groupby("time")["sla"].mean().reset_index()
    title = f"TML Rata-Rata Nasional Tahun {tahun}"
    fig = px.line(df_grouped, x="time", y="sla", title=title, labels={"sla": "Tinggi Muka Laut (m)", "time": "Bulan"})
    return (fig, df_grouped) if return_df else fig

def plot_proyeksi_tml_tahunan(tahun, return_df=False):
    df = load_timeseries_data('proyeksi')
    df_tahun = df[df["year"] == int(tahun)]
    if df_tahun.empty: return (None, None) if return_df else None
    df_grouped = df_tahun.groupby("time")["sla"].mean().reset_index()
    title = f"Proyeksi TML Rata-Rata Nasional Tahun {tahun}"
    fig = px.line(df_grouped, x="time", y="sla", title=title, labels={"sla": "Tinggi Muka Laut (m)", "time": "Bulan"})
    return (fig, df_grouped) if return_df else fig

def _generic_comparison_plot(data_type, level, name1, name2, return_df=False):
    """Fungsi internal generik untuk membandingkan dua wilayah."""
    df_full = load_timeseries_data(data_type)
    df_filtered = df_full[df_full[level].str.lower().isin([name1.lower(), name2.lower()])]
    if df_filtered.empty: return (None, None) if return_df else None

    df_agg = df_filtered.groupby(["time", level])["sla"].mean().reset_index()
    data_type_title = "Proyeksi " if data_type == 'proyeksi' else ""
    title = f"Perbandingan {data_type_title}TML: {name1.title()} vs {name2.title()}"
    fig = px.line(df_agg, x="time", y="sla", color=level, title=title, labels={"sla": "Tinggi Muka Laut (m)", "time": "Waktu"})
    return (fig, df_agg) if return_df else fig

def plot_bandingkan_desa(desa1, desa2, return_df=False):
    return _generic_comparison_plot('observasi', 'desa', desa1, desa2, return_df)

def plot_proyeksi_bandingkan_desa(desa1, desa2, return_df=False):
    return _generic_comparison_plot('proyeksi', 'desa', desa1, desa2, return_df)

def plot_bandingkan_provinsi(provinsi1, provinsi2, return_df=False):
    return _generic_comparison_plot('observasi', 'provinsi', provinsi1, provinsi2, return_df)

def plot_proyeksi_bandingkan_provinsi(provinsi1, provinsi2, return_df=False):
    return _generic_comparison_plot('proyeksi', 'provinsi', provinsi1, provinsi2, return_df)

### ----- FUNGSI RANKING ----- ###

def _generic_ranking_table(data_type, level, text, ascending=False):
    """Fungsi internal generik untuk membuat tabel ranking."""
    df_full = load_timeseries_data(data_type)
    top_n = _extract_top_n(text)
    
    group_cols = ['desa', 'kecamatan', 'kabupaten', 'provinsi'] if level == 'desa' else ['provinsi']
    df_rank = df_full.groupby(group_cols)["sla"].mean().reset_index()
    df_rank = df_rank.sort_values(by="sla", ascending=ascending).head(top_n)
    
    # Formatting
    df_rank.columns = [col.title() for col in df_rank.columns]
    df_rank = df_rank.rename(columns={"Sla": "Rata-rata TML (m)"})
    df_rank["Rata-rata TML (m)"] = df_rank["Rata-rata TML (m)"].round(3)
    df_rank.insert(0, "Peringkat", range(1, len(df_rank) + 1))
    
    return df_rank.reset_index(drop=True)

def ranking_tml_desa(text=None):
    return _generic_ranking_table('observasi', 'desa', text)

def ranking_proyeksi_tml_desa(text=None):
    return _generic_ranking_table('proyeksi', 'desa', text)

def ranking_tml_provinsi(text=None):
    return _generic_ranking_table('observasi', 'provinsi', text)

def ranking_proyeksi_tml_provinsi(text=None):
    return _generic_ranking_table('proyeksi', 'provinsi', text)

# =============================================================================
# 4. FUNGSI PLOTTING PETA (Logikanya berbeda, jadi dipisah)
# =============================================================================

@st.cache_data(show_spinner=False)
def _load_netcdf_data(data_type: str):
    """Memuat data NetCDF untuk peta."""
    if data_type == 'observasi':
        return xr.open_dataset("sea_level_obs.nc")
    elif data_type == 'proyeksi':
        return xr.open_dataset("data/predicted_ssh_monthly_1993-2014_9_canesm5_ssp245_2025-01-16_2100-12-16_indo.nc")
    elif data_type == 'tren_observasi':
        return xr.open_dataset("SSH_trend_indo_1993_2024.nc")
    elif data_type == 'tren_proyeksi':
        return xr.open_dataset("data/SSH_trend_canesm5_ssp245_2025_2100_CNN_LSTM.nc")
    raise ValueError("Tipe data NetCDF tidak valid.")

def _generic_map_plotter(data_type, year=None, return_regions=False):
    """Fungsi generik terpusat untuk membuat semua jenis peta."""
    ds = _load_netcdf_data(data_type)
    
    # Ekstraksi variabel berdasarkan tipe data
    if 'tren' in data_type:
        sla_var = ds['trend']
        title = f"Peta Tren {'Proyeksi ' if 'proyeksi' in data_type else ''}TML"
        hover_text = "Tren TML: {z:.2f} mm/year"
        colorbar_label = "Tren (mm/year)"
        zmin, zmax = -5, 5
    else:
        var_name = 'zos' if 'proyeksi' in data_type else 'sla'
        sla_agg = ds[var_name].groupby("time.year").mean("time")
        if year not in sla_agg.year.values: return (None,)*5 if return_regions else None
        sla_var = sla_agg.sel(year=year)
        title = f"Peta {'Proyeksi ' if 'proyeksi' in data_type else ''}TML Tahun {year}"
        hover_text = "TML: {z:.3f} m"
        colorbar_label = "Tinggi Muka Laut (m)"
        zmin, zmax = -0.25, 0.25

    lat_flat = np.repeat(ds['latitude'].values, len(ds['longitude'].values))
    lon_flat = np.tile(ds['longitude'].values, len(ds['latitude'].values))
    sla_flat = sla_var.values.flatten()

    mask = ~np.isnan(sla_flat)
    lat_valid, lon_valid, sla_valid = lat_flat[mask], lon_flat[mask], sla_flat[mask]

    # Ekstrak info wilayah
    if return_regions:
        max_idx, min_idx = np.argmax(sla_valid), np.argmin(sla_valid)
        region_max = _translate_sea_name(_get_region(IHO_GDF, lat_valid[max_idx], lon_valid[max_idx], ['NAME'], "Wilayah Tidak Diketahui"))
        region_min = _translate_sea_name(_get_region(IHO_GDF, lat_valid[min_idx], lon_valid[min_idx], ['NAME'], "Wilayah Tidak Diketahui"))
        prov_max = _get_region(PROVINSI_GDF, lat_valid[max_idx], lon_valid[max_idx], ['name_id', 'name'], "Jauh dari Daratan")
        prov_min = _get_region(PROVINSI_GDF, lat_valid[min_idx], lon_valid[min_idx], ['name_id', 'name'], "Jauh dari Daratan")
    
    # Pengaturan warna dan binning
    bins = np.linspace(zmin, zmax, 11)
    cmap = cm.get_cmap('coolwarm', 10)
    colors = [to_hex(cmap(i)) for i in range(cmap.N)]
    colors_ext = [to_hex(cmap(0))] + colors + [to_hex(cmap(cmap.N - 1))]
    bin_indices = np.digitize(sla_valid, bins)
    
    # Membuat plot
    fig = go.Figure()
    for i in range(len(colors_ext)):
        mask_i = bin_indices == i
        if not np.any(mask_i): continue
        fig.add_trace(go.Scattermapbox(
            lat=lat_valid[mask_i], lon=lon_valid[mask_i], mode="markers",
            marker=dict(size=3.5, color=colors_ext[i]),
            hoverinfo="text",
            text=[f"Lat: {lt:.2f}<br>Lon: {ln:.2f}<br>" + hover_text.format(z=z) 
                  for lt, ln, z in zip(lat_valid[mask_i], lon_valid[mask_i], sla_valid[mask_i])],
            showlegend=False
        ))

    fig.update_layout(
        title={'text': title, 'x': 0.5},
        mapbox_style="carto-positron",
        mapbox_zoom=3.3,
        mapbox_center={"lat": -2, "lon": 118},
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    
    # (Kode untuk colorbar bisa ditambahkan di sini jika diinginkan)

    if return_regions:
        return fig, region_max, region_min, prov_max, prov_min
    return fig

def peta_tml_tahun(year, return_regions=False): return _generic_map_plotter('observasi', year, return_regions)
def peta_proyeksi_tml_tahun(year, return_regions=False): return _generic_map_plotter('proyeksi', year, return_regions)
def peta_tren_tml_nasional(return_regions=False): return _generic_map_plotter('tren_observasi', None, return_regions)
def peta_tren_proyeksi_tml_nasional(return_regions=False): return _generic_map_plotter('tren_proyeksi', None, return_regions)