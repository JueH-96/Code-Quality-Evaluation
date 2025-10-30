import sys

def calculate_black_area(A, B, C, D):
    def is_black(x, y):
        # Determine the color of the region containing point (x, y)
        x_part = (x > 0) - (x <= 0)
        y_part = (y + 1) // 2 * 2 - y
        diagonal_part = (x + y + 1) // 2 * 2 - (x + y)
        return (x_part + y_part + diagonal_part) % 2 == 1

    # Calculate the total area of the rectangle
    total_area = (C - A) * (D - B)

    # Calculate the area of black regions
    black_area = 0
    for x in range(A, C):
        for y in range(B, D):
            if is_black(x + 0.5, y + 0.5):
                black_area += 1

    # Double the black area
    result = 2 * black_area
    print(result)

# Read input
A, B, C, D = map(int, sys.stdin.readline().split())

# Calculate and print the result
calculate_black_area(A, B, C, D)