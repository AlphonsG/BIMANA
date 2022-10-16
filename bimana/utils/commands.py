def parse_input_bgr(
    input_bgr: tuple[str, str, str],
) -> tuple[int | float, int | float, int | float]:
    """Parses input BGR pixel values.

    Converts input BGR values from strings (of raw pixel values between 0 and
    255) to integers or from strings (of pixel intensity values between 0.0
    and 1.0) to floats.

    Args:
        input_bgr: A sequence of BGR values as strings.

    Returns:
        A sequence of parsed BGR pixel values.
    """
    output_bgr = []
    for val in input_bgr:
        try:
            output_bgr.append(int(val))
        except ValueError:
            output_bgr.append(float(val))

    return tuple(output_bgr)
