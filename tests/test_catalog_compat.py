from oasis import compat as C
from oasis.catalog import classify_measure, classify_unit, normalize_label


def test_unit_classification():
    assert classify_unit("%")[1] == "terkonfirmasi_kamus"
    assert classify_unit("Rp M (sesuai label sumber)")[:2] == ("Rp M", "label_sumber")
    assert classify_unit("Tidak tercantum pada header sumber")[1] == "belum_terkonfirmasi"
    assert classify_unit("Tanggal (YYYYMMDD)")[1] == "kunci"


def test_measure_classification():
    assert classify_measure("Total aset pada akhir periode.", "Rp M", "label_sumber", "Total Aset")[0] == "stok"
    assert classify_measure("Total volume transaksi saham selama periode.", "Miliar saham", "x", "Volume")[0] == "arus"
    m = classify_measure("Risk Based Capital (RBC) ...", None, "belum_terkonfirmasi", "RBC Asuransi Jiwa")
    assert m[0] == "belum_pasti" and m[1] is False


def test_normalize_label():
    assert normalize_label("Combined Rasio  Asuransi Jiwa", "Perusahaan Asuransi Jiwa") == "combined rasio"
    assert normalize_label("Total Aset (Rp M)", "Bank Umum") == "total aset"


def test_rbc_unit_unconfirmed(demo_ds):
    rbc = demo_ds.catalog[demo_ds.catalog["concept_id"] == "rbc"]
    assert (rbc["unit_status"] == "belum_terkonfirmasi").all()


def test_compat_levels(demo_ds):
    info = demo_ds.info
    assert C.pair(info("bank_umum::rasio_npl"), info("bpr::rasio_npl"))["level"] == "absolut"
    assert C.pair(info("bank_umum::rasio_npl"), info("bank_umum::rasio_car"))["level"] == "perubahan"
    aj, au = "perusahaan_asuransi_jiwa::total_aset_asuransi_jiwa", "perusahaan_asuransi_umum::total_aset_asuransi_umum"
    assert C.pair(info(aj), info(au))["level"] == "indeks"  # satuan belum terkonfirmasi
    assert C.pair(info("bank_umum::total_kredit_rp_m"), info("bank_umum::rasio_npl"))["level"] == "berdampingan"
    # setelah konfirmasi satuan oleh pengguna → absolut
    ds2 = demo_ds.with_unit_confirmations({aj: {"unit": "Rp miliar", "oleh": "uji", "dasar": "uji"},
                                           au: {"unit": "Rp miliar", "oleh": "uji", "dasar": "uji"}})
    assert C.pair(ds2.info(aj), ds2.info(au))["level"] == "absolut"


def test_kapitalisasi_vs_aset_no_conversion(demo_ds):
    a = demo_ds.info("pasar_modal::kapitalisasi_pasar_rp_triliun")
    b = demo_ds.info("bank_umum::total_aset_rp_m")
    p = C.pair(a, b)
    assert p["level"] != "absolut" and "tanpa aturan konversi" in p["alasan"]
