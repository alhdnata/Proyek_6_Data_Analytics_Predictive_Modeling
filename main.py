
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
DATA_FILE = DATA_DIR / "penjualan_bulanan.csv"


def buat_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        return

    data = {
        "bulan": pd.date_range(
            start="2025-01-01",
            periods=18,
            freq="MS"
        ),
        "penjualan": [
            120, 128, 135, 132, 145, 153,
            160, 158, 172, 180, 188, 195,
            202, 215, 220, 230, 238, 250
        ]
    }

    pd.DataFrame(data).to_csv(DATA_FILE, index=False)
    print("Dataset penjualan contoh berhasil dibuat.")


def muat_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["bulan"])

    df["penjualan"] = pd.to_numeric(
        df["penjualan"], errors="coerce"
    )

    df = df.dropna(
        subset=["bulan", "penjualan"]
    )

    df = df[df["penjualan"] >= 0]
    df = df.sort_values("bulan")
    df = df.drop_duplicates(subset=["bulan"])

    if len(df) < 6:
        raise ValueError(
            "Minimal diperlukan 6 bulan data untuk analisis."
        )

    df["nomor_bulan"] = range(1, len(df) + 1)
    df["pertumbuhan_persen"] = (
        df["penjualan"].pct_change() * 100
    )

    return df


def analisis_data(df):
    print("\n" + "=" * 50)
    print("ANALISIS DATA PENJUALAN BULANAN")
    print("=" * 50)

    total = df["penjualan"].sum()
    rata_rata = df["penjualan"].mean()
    tertinggi = df.loc[df["penjualan"].idxmax()]
    terendah = df.loc[df["penjualan"].idxmin()]

    print(f"Jumlah periode       : {len(df)} bulan")
    print(f"Total penjualan      : {total:,.2f}")
    print(f"Rata-rata per bulan  : {rata_rata:,.2f}")
    print(
        f"Penjualan tertinggi  : {tertinggi['penjualan']:,.2f}"
        f" ({tertinggi['bulan'].strftime('%B %Y')})"
    )
    print(
        f"Penjualan terendah   : {terendah['penjualan']:,.2f}"
        f" ({terendah['bulan'].strftime('%B %Y')})"
    )

    print("\nData dan pertumbuhan bulanan:")
    print(
        df[
            ["bulan", "penjualan", "pertumbuhan_persen"]
        ].to_string(
            index=False,
            formatters={
                "bulan": lambda x: x.strftime("%Y-%m"),
                "penjualan": lambda x: f"{x:,.2f}",
                "pertumbuhan_persen": (
                    lambda x: "-" if pd.isna(x)
                    else f"{x:.2f}%"
                )
            }
        )
    )


def latih_dan_prediksi(df):
    X = df[["nomor_bulan"]]
    y = df["penjualan"]

    # Data terakhir dipakai untuk menguji model.
    jumlah_uji = max(2, round(len(df) * 0.2))
    X_train = X.iloc[:-jumlah_uji]
    X_test = X.iloc[-jumlah_uji:]
    y_train = y.iloc[:-jumlah_uji]
    y_test = y.iloc[-jumlah_uji:]

    model_uji = LinearRegression()
    model_uji.fit(X_train, y_train)

    prediksi_uji = model_uji.predict(X_test)

    mae = mean_absolute_error(y_test, prediksi_uji)
    rmse = mean_squared_error(
        y_test, prediksi_uji
    ) ** 0.5

    print("\n" + "=" * 50)
    print("EVALUASI MODEL REGRESI LINEAR")
    print("=" * 50)
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

    # Latih kembali menggunakan semua data yang tersedia.
    model = LinearRegression()
    model.fit(X, y)

    bulan_selanjutnya = len(df) + 1
    prediksi_baru = model.predict(
        pd.DataFrame(
            {"nomor_bulan": [bulan_selanjutnya]}
        )
    )[0]

    tanggal_prediksi = (
        df["bulan"].max() + pd.offsets.MonthBegin(1)
    )

    print("\nPREDIKSI BULAN BERIKUTNYA")
    print("Bulan:", tanggal_prediksi.strftime("%B %Y"))
    print(f"Estimasi penjualan: {prediksi_baru:,.2f}")

    hasil_uji = pd.DataFrame({
        "bulan": df["bulan"].iloc[-jumlah_uji:].values,
        "aktual": y_test.values,
        "prediksi_model": prediksi_uji
    })

    hasil_uji["selisih_absolut"] = (
        hasil_uji["aktual"] -
        hasil_uji["prediksi_model"]
    ).abs()

    hasil_uji.to_csv(
        OUTPUT_DIR / "evaluasi_model.csv",
        index=False
    )

    hasil_prediksi = pd.DataFrame([{
        "bulan_prediksi": tanggal_prediksi.strftime("%Y-%m"),
        "estimasi_penjualan": round(float(prediksi_baru), 2),
        "mae_uji": round(float(mae), 2),
        "rmse_uji": round(float(rmse), 2)
    }])

    hasil_prediksi.to_csv(
        OUTPUT_DIR / "prediksi_bulan_berikutnya.csv",
        index=False
    )

    return model, prediksi_baru, tanggal_prediksi, hasil_uji


def buat_grafik(df, model, prediksi_baru, tanggal_prediksi):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    nomor_bulan = df[["nomor_bulan"]]
    garis_tren = model.predict(nomor_bulan)

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 9)
    )

    axes[0].plot(
        df["bulan"], df["penjualan"],
        marker="o", label="Penjualan aktual"
    )
    axes[0].plot(
        df["bulan"], garis_tren,
        linestyle="--", label="Tren regresi"
    )
    axes[0].scatter(
        [tanggal_prediksi], [prediksi_baru],
        marker="X", s=120, label="Prediksi berikutnya"
    )
    axes[0].set_title("Tren dan Prediksi Penjualan")
    axes[0].set_xlabel("Bulan")
    axes[0].set_ylabel("Penjualan")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].bar(
        df["bulan"].dt.strftime("%Y-%m"),
        df["pertumbuhan_persen"]
    )
    axes[1].axhline(0, linewidth=1)
    axes[1].set_title("Pertumbuhan Penjualan Bulanan")
    axes[1].set_xlabel("Bulan")
    axes[1].set_ylabel("Pertumbuhan (%)")
    axes[1].tick_params(
        axis="x", rotation=45
    )
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "analisis_penjualan.png",
        dpi=160,
        bbox_inches="tight"
    )
    plt.show()


def main():
    print("=" * 50)
    print("DATA ANALYTICS & PREDICTIVE MODELING")
    print("=" * 50)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    buat_dataset()
    df = muat_data()

    analisis_data(df)

    model, prediksi, tanggal, hasil_uji = (
        latih_dan_prediksi(df)
    )

    buat_grafik(df, model, prediksi, tanggal)

    df.to_csv(
        OUTPUT_DIR / "data_analisis.csv",
        index=False
    )

    print("\nProgram selesai.")
    print("Periksa folder output untuk seluruh hasil.")


if __name__ == "__main__":
    main()
