from bimana.utils.commands import parse_input_bgr


def test_parse_input_bgr() -> None:
    input_bgr = ('0.3', '255', '0.0')
    output_bgr = parse_input_bgr(input_bgr)
    expected_result = (0.3, 255, 0.0)

    assert output_bgr == expected_result
