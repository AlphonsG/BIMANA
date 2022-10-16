import numpy as np

from bimana.utils.quantification import percentage_foreground_in_region


def test_percentage_foreground_in_region() -> None:
    hw = (10, 20)

    img = np.zeros(hw, bool)
    img[5:, 10:] = 1

    region = np.zeros(hw, bool)
    region[5:, :] = 1

    pct, fg_ct, reg_cnt = percentage_foreground_in_region(img, region)
    exp_pct, exp_fg_cnt, exp_reg_cnt = 50, 50, 100

    assert (pct, fg_ct, reg_cnt) == (exp_pct, exp_fg_cnt, exp_reg_cnt)
