from pathlib import Path

import numpy as np
import pytest
from core.data import load_iq_csv


def test_load_iq_csv_builds_complex_input_and_output(tmp_path: Path) -> None:
    csv = tmp_path / "pa.csv"
    csv.write_text("Xreal,Ximg,Yreal,Yimg\n1,2,10,20\n3,4,30,40\n")

    x, y = load_iq_csv(csv)

    np.testing.assert_allclose(x, [1 + 2j, 3 + 4j])
    np.testing.assert_allclose(y, [10 + 20j, 30 + 40j])


def test_load_iq_csv_builds_complex_input_and_output_with_three_columns(
    tmp_path: Path,
) -> None:
    csv = tmp_path / "pa.csv"
    csv.write_text("Xreal,Ximg,Yreal\n1,2,10\n3,4,30\n")

    with pytest.raises(ValueError, match="4"):
        load_iq_csv(csv)
