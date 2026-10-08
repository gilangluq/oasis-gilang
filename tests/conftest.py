import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEMO = ROOT / "data" / "demo" / "oasis_demo_simulasi.xlsx"
REAL = os.environ.get("OASIS_TEST_WORKBOOK") or str(ROOT / "data" / "public" / "Dataset_dengan_Metadata_dan_Kamus_Data.xlsx")


@pytest.fixture(scope="session")
def demo_ds():
    from oasis.data_prep import load_workbook
    if not DEMO.exists():
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_demo_workbook.py")], check=True)
    return load_workbook(str(DEMO), DEMO.name, status_data="Simulasi")


@pytest.fixture(scope="session")
def real_ds():
    if not REAL or not Path(REAL).exists():
        pytest.skip("Set OASIS_TEST_WORKBOOK=<path workbook asli> untuk menjalankan uji ini.")
    from oasis.data_prep import load_workbook
    return load_workbook(REAL, Path(REAL).name)


@pytest.fixture(scope="session")
def parser(demo_ds):
    from oasis.nlu import QuestionParser
    return QuestionParser(demo_ds.catalog)
