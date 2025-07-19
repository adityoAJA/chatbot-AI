
def generate_narrative(tag, **kwargs):
    
    if tag == "plot_tml_desa":
        df = kwargs.get("df")
        desa = kwargs.get("desa", "N/A").title()
        if df is None or df.empty: return f"Maaf, data untuk Desa {desa} tidak dapat ditemukan."
        avg_val = df['sla'].mean()
        max_row = df.loc[df['sla'].idxmax()]
        min_row = df.loc[df['sla'].idxmin()]
        rentang = max_row['sla'] - min_row['sla']
        return (
            f"Grafik ini menunjukkan catatan tinggi muka laut (TML) di **Desa {desa}**. Sepanjang periode yang diamati, rata-rata ketinggian air laut adalah **{avg_val:.2f} meter**. "
            f"Ketinggian air laut pernah mencapai puncaknya di **{max_row['sla']:.2f} meter** pada bulan {max_row['time'].strftime('%B %Y')}, dan titik terendahnya di **{min_row['sla']:.2f} meter** pada {min_row['time'].strftime('%B %Y')}. "
            f"Adanya rentang naik-turun air setinggi **{rentang:.2f} meter** dalam setahun menunjukkan adanya pengaruh musim yang kuat, seperti musim angin barat dan timur, yang wajar terjadi di wilayah pesisir."
        )

    elif tag == "plot_tml_tahunan":
        df = kwargs.get("df")
        tahun = kwargs.get("tahun")
        if df is None or df.empty:
            return f"Data TML untuk tahun {tahun} tidak tersedia."

        bulan_tertinggi = df.loc[df['sla'].idxmax()]
        bulan_terendah = df.loc[df['sla'].idxmin()]
        rata2 = df['sla'].mean()

        return (
            f"Pada tahun **{tahun}**, tinggi muka laut rata-rata nasional per bulan menunjukkan variasi yang cukup jelas. "
            f"Bulan dengan TML tertinggi adalah **bulan {(bulan_tertinggi['time'].strftime('%B'))}** ({bulan_tertinggi['sla']:.2f} meter), "
            f"sedangkan terendah terjadi pada bulan **{(bulan_terendah['time'].strftime('%B'))}** ({bulan_terendah['sla']:.2f} meter). "
            f"Secara umum, rata-rata tahunan adalah **{rata2:.2f} meter**."
        )

    elif tag == "bandingkan_desa":
        desa1 = kwargs.get("desa1", "A").title()
        desa2 = kwargs.get("desa2", "B").title()
        df = kwargs.get("df")

        if df is None or df.empty:
            return f"Data tidak tersedia untuk membandingkan **{desa1}** dan **{desa2}**."

        # Normalisasi nama desa
        df["desa"] = df["desa"].str.lower()
        d1 = desa1.lower()
        d2 = desa2.lower()

        # Hitung rata-rata keseluruhan
        rata1 = df[df["desa"] == d1]["sla"].mean()
        rata2 = df[df["desa"] == d2]["sla"].mean()

        # Bandingkan mana yang lebih tinggi
        if rata1 > rata2:
            lebih_tinggi = desa1
            lebih_rendah = desa2
        else:
            lebih_tinggi = desa2
            lebih_rendah = desa1

        return (
            f"Grafik ini membandingkan perubahan tinggi muka laut (TML) selama periode **1993–2023** antara **Desa {desa1}** dan **Desa {desa2}**.\n\n"
            f"Secara keseluruhan, perubahan tinggi muka laut di **{lebih_tinggi}** lebih tinggi dibandingkan dengan **{lebih_rendah}**.\n\n"
            "Perbedaan ini dapat dipengaruhi oleh kondisi geografis lokal seperti morfologi pantai, sedimentasi, atau faktor antropogenik seperti penurunan muka tanah."
        )
    
    elif tag == "bandingkan_provinsi":
        provinsi1 = kwargs.get("provinsi1", "A").title()
        provinsi2 = kwargs.get("provinsi2", "B").title()
        df = kwargs.get("df")

        if df is None or df.empty:
            return f"Data tidak tersedia untuk membandingkan **{provinsi1}** dan **{provinsi2}**."

        # Normalisasi nama desa
        df["provinsi"] = df["provinsi"].str.lower()
        d1 = provinsi1.lower()
        d2 = provinsi2.lower()

        # Hitung rata-rata keseluruhan
        rata1 = df[df["provinsi"] == d1]["sla"].mean()
        rata2 = df[df["provinsi"] == d2]["sla"].mean()

        # Bandingkan mana yang lebih tinggi
        if rata1 > rata2:
            lebih_tinggi = provinsi1
            lebih_rendah = provinsi2
        else:
            lebih_tinggi = provinsi2
            lebih_rendah = provinsi1

        return (
            f"Grafik ini membandingkan perubahan tinggi muka laut (TML) selama periode **1993–2023** antara **Provinsi {provinsi1}** dan **Provinsi {provinsi2}**.\n\n"
            f"Secara keseluruhan, perubahan tinggi muka laut di **{lebih_tinggi}** lebih tinggi dibandingkan dengan **{lebih_rendah}**.\n\n"
            "Perbedaan ini dapat dipengaruhi oleh kondisi geografis lokal seperti morfologi pantai, sedimentasi, atau faktor antropogenik seperti penurunan muka tanah."
        )

    elif "tren_tml" in tag: # Mencakup tren desa, kecamatan, kabupaten, dan nasional
        trend_val = kwargs.get("trend")
        if trend_val is None: return "Tren tidak dapat dihitung."
        
        lokasi = "di tingkat Nasional"
        if "desa" in tag: lokasi = f"di Desa {kwargs.get('desa', 'N/A').title()}"
        elif "kecamatan" in tag: lokasi = f"di Kecamatan {kwargs.get('kecamatan', 'N/A').title()}"
        elif "kabupaten" in tag: lokasi = f"di Kabupaten {kwargs.get('kabupaten', 'N/A').title()}"
        
        kenaikan_30 = trend_val * 0.3
        return (
            f"{lokasi.title()}, tercatat ada tren kenaikan permukaan air laut sekitar **{abs(trend_val):.2f} milimeter setiap tahun**. "
            f"Ini berarti dalam 30 tahun terakhir, permukaan air laut di wilayah ini telah bertambah tinggi sekitar **{kenaikan_30:.1f} centimeter**. "
            "Kenaikan yang terus-menerus ini menjadi alasan utama mengapa genangan air pasang (rob) bisa menjadi lebih sering atau lebih luas dari tahun ke tahun, sehingga perlu menjadi perhatian bagi warga dan pemerintah setempat."
        )

    elif "ranking_tml" in tag: # Mencakup ranking desa dan provinsi
        df = kwargs.get("df")
        level = "Provinsi" if "Provinsi" in df.columns else "Desa"
        if df is None or df.empty: return f"Peringkat {level} tidak dapat dibuat."
        
        tertinggi = df.iloc[0]
        rata_rata_top_n = df["Rata-rata TML (m)"].mean()
        return (
            f"Berikut adalah daftar {level} dengan rata-rata TML paling tinggi di Indonesia. **{tertinggi[level]}** tercatat sebagai wilayah dengan TML tertinggi, yaitu **{tertinggi['Rata-rata TML (m)']:.2f} meter**. "
            f"Sebagai perbandingan, rata-rata TML dari semua wilayah di daftar ini adalah **{rata_rata_top_n:.2f} meter**. {level} yang berada di peringkat atas secara alami lebih rentan terhadap dampak naiknya air laut, seperti abrasi dan intrusi air asin ke sumur warga."
        )

    elif tag == "peta_tml_tahun":
        tahun = kwargs.get("tahun")
        region_max = kwargs.get("region_max", "N/A")
        return (
            f"Peta ini seperti foto satelit yang menunjukkan kondisi TML pada tahun **{tahun}**. Warna merah berarti permukaan airnya sedikit lebih tinggi dari rata-rata, sedangkan biru lebih rendah. "
            f"Kita bisa lihat ada pola tertentu, di mana perairan seperti **{region_max}** cenderung 'menggembung' atau memiliki permukaan yang lebih tinggi. "
            "Hal ini bisa disebabkan oleh banyak faktor, termasuk suhu air laut yang lebih hangat di wilayah tersebut atau karena pengaruh arus laut."
        )

    elif tag == "peta_tren_tml_nasional":
        region_max = kwargs.get("region_max", "N/A")
        prov_max = kwargs.get("prov_max", "N/A")
        return (
            f"Peta ini adalah potret laju kenaikan air laut di seluruh perairan Indonesia. Warna merah menunjukkan laju kenaikan yang lebih cepat, sementara warna biru menunjukkan laju yang lebih lambat atau stabil. "
            f"Dapat dilihat bahwa beberapa wilayah, seperti perairan di sekitar **{region_max}** (dekat Provinsi **{prov_max}**), menjadi 'hotspot' atau titik panas dengan laju kenaikan tercepat. "
            "Informasi ini sangat penting bagi pemerintah untuk memprioritaskan wilayah mana yang paling mendesak untuk mendapatkan program perlindungan pantai dan adaptasi masyarakat."
        )

    # Di dalam functions/narrative.py
    elif "grafik_tahunan" in tag:
        tahun = kwargs.get("tahun")
        level = "Kabupaten" if "kabupaten" in tag else "Kecamatan"
        nama_wilayah = kwargs.get(level.lower(), "N/A").title()
        df = kwargs.get("df")
        if df is None or df.empty: return f"Data untuk {level} {nama_wilayah} tidak ditemukan."
        
        max_row = df.loc[df['sla'].idxmax()]
        min_row = df.loc[df['sla'].idxmin()]
        avg_val = df['sla'].mean()
        return (
            f"Grafik ini menunjukkan variasi rata-rata tinggi muka laut bulanan di **{level} {nama_wilayah}** selama tahun **{tahun}**. "
            f"Puncak ketinggian air laut terjadi pada bulan **{max_row['time'].strftime('%B')}** ({max_row['sla']:.2f} m), sedangkan titik terendah terjadi pada **{min_row['time'].strftime('%B')}** ({min_row['sla']:.2f} m). "
            f"Rata-rata tinggi muka laut tahunan untuk seluruh wilayah ini adalah **{avg_val:.2f} meter**."
        )

    return "Berikut adalah visualisasi data tinggi muka laut yang diminta. Grafik ini dapat membantu Anda memahami kondisi di wilayah tersebut."