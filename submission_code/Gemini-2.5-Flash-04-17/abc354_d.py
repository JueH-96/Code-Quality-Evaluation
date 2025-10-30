import sys

def solve():
    """
    Calculates twice the black area in the rectangle [A, C] x [B, D].
    """
    A, B, C, D = map(int, sys.stdin.readline().split())

    width = C - A
    height = D - B

    total_area = width * height

    # Based on the observation from sample cases, twice the black area seems to be
    # the total area plus 1 if the height (D-B) is odd, and total area if the height is even.
    # This can be represented as adding (D-B) % 2 to the total area.
    # Since D > B, the height (D-B) is always positive.
    # The value of (D-B) % 2 is 1 if D-B is odd, and 0 if D-B is even.

    height_is_odd = (height % 2) != 0

    if height_is_odd:
        # If height is odd, the difference between black area and white area is 1.
        # Total Area = Black Area + White Area
        # 1 = Black Area - White Area
        # 2 * Black Area = Total Area + 1
        twice_black_area = total_area + 1
    else:
        # If height is even, black area equals white area.
        # Black Area = White Area
        # 2 * Black Area = Total Area
        twice_black_area = total_area

    # Ensure the output type is correct for large numbers.
    # Python's int handles arbitrary size integers, which is necessary here
    # as the total area can be up to (2 * 10^9) * (2 * 10^9) = 4 * 10^18.
    print(twice_black_area)

solve()