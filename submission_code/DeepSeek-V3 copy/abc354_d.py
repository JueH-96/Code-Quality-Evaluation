def calculate_black_area(A, B, C, D):
    def get_color(x, y):
        # Determine the color based on the rules
        # The region containing (0.5, 0.5) is black
        # The color alternates based on the lines x = n, y = 2n, x + y = 2n
        # We can use the parity of floor(x) + floor(y/2) + floor((x + y)/2)
        # But a simpler approach is to use the parity of floor(x) + floor(y) + floor((x + y)/2)
        # However, the correct approach is to consider the parity of floor(x) + floor(y/2) + floor((x + y)/2)
        # Because y = 2n and x + y = 2n are the lines that affect the color
        # So, the color is black if (floor(x) + floor(y / 2) + floor((x + y) / 2)) % 2 == 0
        # Since (0.5, 0.5) is black, we need to adjust the formula to match
        # For (0.5, 0.5): floor(0.5) + floor(0.25) + floor(0.5 + 0.25) = 0 + 0 + 0 = 0 → black
        # So the formula is correct
        return (int(x) + int(y // 2) + int((x + y) // 2)) % 2 == 0

    # To calculate the area, we need to count the number of black cells in the rectangle
    # Since the pattern repeats every 2 units in x and y, we can use a grid approach
    # But for large ranges, we need a mathematical approach

    # The area of black regions can be computed as follows:
    # The total area is (C - A) * (D - B)
    # The black area is half of the total area, plus or minus some adjustments based on the boundaries

    # The pattern is such that in every 2x2 square, there are 2 black and 2 white cells
    # So, the black area is approximately half of the total area

    # However, the exact calculation depends on the boundaries

    # We can compute the number of black cells in the rectangle by considering the parity of the coordinates

    # Let's compute the number of black cells in the rectangle

    # The number of black cells is:
    # (number of cells where (x + y) is even) / 2 + (number of cells where (x + y) is odd) / 2
    # But this is not directly helpful

    # Instead, we can use the fact that the black area is half the total area, plus or minus the boundary effects

    # The total area is (C - A) * (D - B)
    total_area = (C - A) * (D - B)

    # The black area is total_area / 2, plus or minus some adjustments based on the parity of the boundaries

    # To compute the adjustments, we need to consider the parity of the coordinates at the boundaries

    # The adjustment is:
    # (number of black cells on the left and bottom boundaries) - (number of white cells on the left and bottom boundaries)

    # But this is complex to compute directly

    # Instead, we can use the following approach:
    # The black area is (total_area + parity_adjustment) / 2

    # The parity_adjustment is:
    # (number of black cells at the corners) - (number of white cells at the corners)

    # So, we need to compute the color at the four corners of the rectangle

    corners = [
        (A, B),
        (A, D),
        (C, B),
        (C, D)
    ]

    black_corners = 0
    for (x, y) in corners:
        if get_color(x, y):
            black_corners += 1

    # The parity_adjustment is black_corners - (4 - black_corners) = 2 * black_corners - 4
    parity_adjustment = 2 * black_corners - 4

    # The black area is (total_area + parity_adjustment) / 2
    black_area = (total_area + parity_adjustment) // 2

    # Since the problem asks for twice the black area, we return 2 * black_area
    return 2 * black_area

# Read input
A, B, C, D = map(int, input().split())

# Compute and print the result
print(calculate_black_area(A, B, C, D))