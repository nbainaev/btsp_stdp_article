"""Optional original-source fixture for the two parity checks."""
import importlib.util
import shutil
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def legacy_reference(tmp_path):
    source = ROOT / 'reference' / 'closed_loop_2026_11_16.py'
    params = ROOT / 'reference' / 'A2_SECONDS_STRICT.json'
    if not source.is_file() or not params.is_file():
        pytest.skip('optional original reference and canonical parameter fixture are unavailable')
    (tmp_path / 'reference').mkdir()
    (tmp_path / 'configs').mkdir()
    copied = tmp_path / 'reference' / source.name
    shutil.copyfile(source, copied)
    shutil.copyfile(params, tmp_path / 'configs' / params.name)
    spec = importlib.util.spec_from_file_location('btsp_legacy_reference', copied)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
