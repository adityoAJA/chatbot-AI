def generate_narrative(tag, **kwargs):

    if tag == "plot_proyeksi_tml_desa":
        df = kwargs.get("df")
        desa = kwargs.get("desa", "N/A").title()
        if df is None or df.empty: return f"Maaf, data proyeksi untuk Desa {desa} tidak dapat dibuat."
        avg_val = df['sla'].mean()
        max_row = df.loc[df['sla'].idxmax()]
        min_row = df.loc[df['sla'].idxmin()]
        rentang_proyeksi = max_row['sla'] - min_row['sla']
        return (
            f"Ini adalah perkiraan tinggi muka laut (TML) di **Desa {desa}** untuk masa depan. Berdasarkan skenario iklim, rata-rata ketinggian air laut diperkirakan sekitar **{avg_val:.2f} meter**. "
            f"Puncaknya diproyeksikan bisa mencapai **{max_row['sla']:.2f} meter** pada {max_row['time'].strftime('%B %Y')}, dengan titik terendah sekitar **{min_row['sla']:.2f} meter**. "
            f"Adanya potensi rentang naik-turun air setinggi **{rentang_proyeksi:.2f} meter** dalam setahun memberikan gambaran kemungkinan pasang surut ekstrem yang perlu diantisipasi."
        )

    elif tag == "plot_proyeksi_tml_tahunan":
        df = kwargs.get("df")
        tahun = kwargs.get("tahun")
        if df is None or df.empty:
            return f"Data TML untuk tahun {tahun} tidak tersedia."

        bulan_tertinggi = df.loc[df['sla'].idxmax()]
        bulan_terendah = df.loc[df['sla'].idxmin()]
        rata2 = df['sla'].mean()

        return (
            f"Pada tahun **{tahun}**, proyeksi tinggi muka laut rata-rata nasional per bulan menunjukkan variasi yang cukup jelas. "
            f"Bulan dengan TML tertinggi adalah **bulan {(bulan_tertinggi['time'].strftime('%B'))}** ({bulan_tertinggi['sla']:.2f} meter), "
            f"sedangkan terendah terjadi pada bulan **{(bulan_terendah['time'].strftime('%B'))}** ({bulan_terendah['sla']:.2f} meter). "
            f"Secara umum, rata-rata tahunan adalah **{rata2:.2f} meter**."
        )

    elif tag == "proyeksi_bandingkan_desa":
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
            f"Grafik ini membandingkan proyeksi perubahan tinggi muka laut selama periode **2025–2100** antara **Desa {desa1}** dan **Desa {desa2}**.\n\n"
            f"Secara keseluruhan, perubahan tinggi muka laut di **{lebih_tinggi}** lebih tinggi dibandingkan dengan **{lebih_rendah}**.\n\n"
            "Perbedaan ini dapat dipengaruhi oleh kondisi geografis lokal seperti morfologi pantai, sedimentasi, atau faktor antropogenik seperti penurunan muka tanah."
        )
    
    elif tag == "proyeksi_bandingkan_provinsi":
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
            f"Grafik ini membandingkan proyeksi perubahan tinggi muka laut (TML) selama periode **2025–2100** antara **Provinsi {provinsi1}** dan **Provinsi {provinsi2}**.\n\n"
            f"Secara keseluruhan, perubahan tinggi muka laut di **{lebih_tinggi}** lebih tinggi dibandingkan dengan **{lebih_rendah}**.\n\n"
            "Perbedaan ini dapat dipengaruhi oleh kondisi geografis lokal seperti morfologi pantai, sedimentasi, atau faktor antropogenik seperti penurunan muka tanah."
        )

    elif "tren_proyeksi" in tag: # Mencakup tren desa, kecamatan, kabupaten, dan nasional
        trend_val = kwargs.get("trend")
        if trend_val is None: return "Tren proyeksi tidak dapat dihitung."
        
        lokasi = "di tingkat Nasional"
        if "desa" in tag: lokasi = f"di Desa {kwargs.get('desa', 'N/A').title()}"
        elif "kecamatan" in tag: lokasi = f"di Kecamatan {kwargs.get('kecamatan', 'N/A').title()}"
        elif "kabupaten" in tag: lokasi = f"di Kabupaten {kwargs.get('kabupaten', 'N/A').title()}"
        
        proyeksi_100 = trend_val * 0.1
        return (
            f"{lokasi.title()}, tren proyeksi menunjukkan potensi kenaikan permukaan air laut sebesar **{abs(trend_val):.2f} milimeter setiap tahunnya**. "
            f"Artinya, jika tidak ada aksi mitigasi iklim yang signifikan, dalam 100 tahun ke depan permukaan air laut di wilayah ini **berpotensi bertambah tinggi sekitar {proyeksi_100:.1f} meter**. "
            "Angka ini menjadi dasar penting bagi perencanaan jangka panjang, seperti menentukan lokasi pembangunan infrastruktur baru atau merancang sistem drainase yang tahan terhadap genangan rob di masa depan."
        )

    elif "ranking_proyeksi" in tag: # Mencakup ranking desa dan provinsi
        df = kwargs.get("df")
        level = "Provinsi" if "Provinsi" in df.columns else "Desa"
        if df is None or df.empty: return f"Peringkat proyeksi {level} tidak dapat dibuat."
        
        tertinggi = df.iloc[0]
        terendah = df.iloc[-1]
        selisih = tertinggi['Rata-rata TML (m)'] - terendah['Rata-rata TML (m)']
        return (
            f"Berdasarkan proyeksi, berikut adalah daftar {level} yang diperkirakan akan memiliki TML paling tinggi. **{tertinggi[level]}** diproyeksikan memiliki TML tertinggi, yaitu **{tertinggi['Rata-rata TML (m)']:.2f} meter**. "
            f"Adanya potensi selisih yang mencapai **{selisih:.2f} meter** dengan {level} lain menunjukkan adanya potensi ketimpangan dampak perubahan iklim. Wilayah di peringkat atas ini adalah prioritas untuk program adaptasi masa depan."
        )

    elif tag == "peta_proyeksi_tml_tahun":
        tahun = kwargs.get("tahun")
        region_max = kwargs.get("region_max", "N/A")
        return (
            f"Peta ini adalah sebuah skenario atau perkiraan kondisi TML di masa depan pada tahun **{tahun}**. Warna merah menunjukkan wilayah yang diproyeksikan memiliki permukaan air lebih tinggi dari rata-rata. "
            f"Wilayah perairan seperti **{region_max}** diperkirakan akan menjadi area dengan TML yang relatif sangat tinggi. "
            "Peta proyeksi seperti ini digunakan oleh para perencana dan ilmuwan untuk memodelkan dampak perubahan iklim dan merancang strategi pencegahannya."
        )

    elif tag == "peta_tren_proyeksi_tml_nasional":
        region_max = kwargs.get("region_max", "N/A")
        prov_max = kwargs.get("prov_max", "N/A")
        return (
            f"Peta ini menggambarkan perkiraan laju kenaikan air laut di masa depan untuk seluruh Indonesia. Ini berfungsi sebagai 'peringatan dini' untuk menunjukkan wilayah mana yang paling berisiko. "
            f"Wilayah yang berwarna merah pekat, seperti di perairan **{region_max}** (dekat Provinsi **{prov_max}**), diproyeksikan akan mengalami laju kenaikan tercepat. "
            "Artinya, masyarakat dan pemerintah di daerah tersebut perlu mempersiapkan rencana adaptasi yang lebih serius, misalnya dengan membangun tanggul laut, menanam mangrove, atau merelokasi aset-aset penting ke tempat yang lebih aman."
        )

    # Di dalam functions/narrative_proj.py
    elif "grafik_proyeksi_tahunan" in tag:
        tahun = kwargs.get("tahun")
        level = "Kabupaten" if "kabupaten" in tag else "Kecamatan"
        nama_wilayah = kwargs.get(level.lower(), "N/A").title()
        df = kwargs.get("df")
        if df is None or df.empty: return f"Data proyeksi untuk {level} {nama_wilayah} tidak ditemukan."
        
        max_row = df.loc[df['sla'].idxmax()]
        min_row = df.loc[df['sla'].idxmin()]
        avg_val = df['sla'].mean()
        return (
            f"Pada masa depan di tahun **{tahun}**, grafik ini memproyeksikan variasi rata-rata tinggi muka laut bulanan di **{level} {nama_wilayah}**. "
            f"Puncak ketinggian air laut diperkirakan terjadi pada bulan **{max_row['time'].strftime('%B')}** ({max_row['sla']:.2f} m), dengan titik terendah pada **{min_row['time'].strftime('%B')}** ({min_row['sla']:.2f} m). "
            f"Rata-rata tinggi muka laut tahunan yang diproyeksikan untuk wilayah ini adalah **{avg_val:.2f} meter**."
        )

    return "Berikut adalah visualisasi proyeksi tinggi muka laut. Data ini adalah perkiraan yang dapat membantu kita bersiap menghadapi masa depan."