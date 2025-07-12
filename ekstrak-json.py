# import xarray as xr
# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import Point
# from tqdm import tqdm

# # === Konfigurasi ===
# nc_file = "hasil/miroc6_ssp585_2021_2100_CNN.nc"  # Ganti dengan nama file NetCDF Anda
# geojson_file = "38_prov_indo_kab.json"
# output_file = "hasil_sla_proj_ekstraksi.xlsx"
# var_name = "zos"

# # === Buka NetCDF dan ubah ke dataframe ===
# print("📦 Membuka file NC...")
# ds = xr.open_dataset(nc_file)

# # Konversi waktu
# print("🕒 Mengubah dimensi waktu...")
# ds["time"] = pd.to_datetime(ds["time"].values, unit="s")

# # Convert ke DataFrame
# print("🧾 Konversi ke DataFrame...")
# df = ds[var_name].to_dataframe().reset_index()

# # Buat GeoDataFrame dari lon/lat
# print("🌍 Mengubah ke GeoDataFrame...")
# geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
# gdf_data = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# # Buka GeoJSON wilayah Indonesia
# print("📑 Membaca file GeoJSON...")
# gdf_admin = gpd.read_file(geojson_file)
# gdf_admin = gdf_admin.to_crs("EPSG:4326")

# # Spatial join
# print("🔗 Spatial join dengan data administratif...")
# gdf_joined = gpd.sjoin(gdf_data, gdf_admin, how="left", predicate="within")

# # Rename dan pilih kolom akhir
# print("📁 Menyusun data akhir...")
# final_df = gdf_joined[["time", "latitude", "longitude", var_name, "WADMKD", "WADMKC", "WADMKK", "WADMPR"]]
# final_df.columns = ["time", "latitude", "longitude", "sla", "desa", "kecamatan", "kabupaten", "provinsi"]

# # Simpan ke Excel
# print(f"💾 Menyimpan hasil ke {output_file}...")
# # final_df.to_excel(output_file, index=False)
# final_df.to_csv("hasil_sla_proj.csv", index=False)
# print("✅ Berhasil disimpan.")

import geopandas as gpd
import xarray as xr
import pandas as pd
from shapely.geometry import Point
import numpy as np

# 1. Load GeoJSON batas admin
gdf = gpd.read_file("chatbot/38_prov_indo_kab.json")

# 2. Load dataset NetCDF laut
ds = xr.open_dataset("chatbot/data/predicted_ssh_monthly_1993-2014_9_canesm5_ssp245_2025-01-16_2100-12-16_indo.nc")
lat = ds.latitude.values
lon = ds.longitude.values
sla = ds.zos  # (time, lat, lon)

# 3. Buat daftar titik (lat, lon)
points = [Point(x, y) for y in lat for x in lon]
df_points = pd.DataFrame(points, columns=["geometry"])
df_points["lon"] = df_points.geometry.apply(lambda p: p.x)
df_points["lat"] = df_points.geometry.apply(lambda p: p.y)

gdf_points = gpd.GeoDataFrame(df_points, geometry="geometry", crs=gdf.crs)

# 4. Spatial Join: cari titik laut yang masuk ke polygon wilayah
joined = gpd.sjoin(gdf_points, gdf, how="inner", predicate="intersects")

# 5. Simpan titik valid
cols_admin = ["WADMPR", "WADMKK", "WADMKC", "WADMKD"]
joined[["lon", "lat", *cols_admin]].to_csv("titik_valid_dengan_admin.csv", index=False)

# 6. Load kembali titik valid
valid_points = pd.read_csv("titik_valid_dengan_admin.csv")

lat_vals = ds.latitude.values
lon_vals = ds.longitude.values

valid_points["lat_idx"] = valid_points["lat"].apply(lambda x: np.abs(lat_vals - x).argmin())
valid_points["lon_idx"] = valid_points["lon"].apply(lambda x: np.abs(lon_vals - x).argmin())

# Ubah ke NumPy array untuk loop lebih cepat (opsional)
lat_idxs = valid_points["lat_idx"].values.astype(int)
lon_idxs = valid_points["lon_idx"].values.astype(int)
lats = valid_points["lat"].values
lons = valid_points["lon"].values
prov = valid_points["WADMPR"].values
kab = valid_points["WADMKK"].values
kec = valid_points["WADMKC"].values
des = valid_points["WADMKD"].values

# 7. Ambil waktu NetCDF (langsung, karena sudah dalam datetime64)
# time_values = pd.to_datetime(ds.time.values) # buka utk data reanalysis / obs
time_values = ds.time.to_index()

# 8. Loop ekstraksi
results = []
missing_counter, valid_counter = 0, 0

for t, timestamp in enumerate(time_values):
    sla_t = sla[t].values
    for i in range(len(lat_idxs)):
        sla_val = sla_t[lat_idxs[i], lon_idxs[i]]
        if np.isnan(sla_val):
            missing_counter += 1
            continue

        valid_counter += 1
        results.append({
            "time": timestamp,
            "latitude": lats[i],
            "longitude": lons[i],
            "sla": sla_val,
            "provinsi": prov[i],
            "kabupaten": kab[i],
            "kecamatan": kec[i],
            "desa": des[i]
        })

# 9. Output
print("📌 ATTRS TIME:", ds.time.attrs)
print("📌 First raw TIME value:", ds.time.values[0])
print("🕓 Waktu awal:", time_values[0])
print("🕓 Waktu akhir:", time_values[-1])
print(f"✅ Total data ditulis: {valid_counter}")
print(f"🚫 Data kosong dilewati: {missing_counter}")

# 10. Simpan CSV
final_df = pd.DataFrame(results)
final_df.to_csv("chatbot/data/canesm5_tml_proj245_by_wilayah.csv", index=False)
print("✅ Ekstraksi selesai dan disimpan.")