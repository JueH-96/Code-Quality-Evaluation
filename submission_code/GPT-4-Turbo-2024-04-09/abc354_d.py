def calculate_black_area(A, B, C, D):
    # Helper function to count the number of lines of a given type within a range [low, high)
    def count_lines(low, high, step):
        # We need to find the first value in the range that is a multiple of `step`
        first = low + (-low % step)
        if first < low:
            first += step
        if first >= high:
            return 0
        # Similarly, find the last value in the range that is a multiple of `step`
        last = high - 1 - ((high - 1) % step)
        if last >= high:
            last -= step
        # Count the number of multiples
        return (last - first) // step + 1

    # Total area of the rectangle
    total_area = (C - A) * (D - B)
    
    # Count vertical lines x = n
    vertical_lines = count_lines(A, C, 1)
    
    # Count horizontal lines y = n where n is even
    horizontal_lines = count_lines(B, D, 2)
    
    # Count diagonal lines x + y = n where n is even
    diagonal_lines = count_lines(A + B, C + D, 2)
    
    # Calculate the number of black cells using inclusion-exclusion principle
    # Initial assumption: half the cells are black
    black_area = total_area / 2
    
    # Adjust based on the counts of lines (each line changes the parity of the regions it borders)
    # Every line type changes the color of the regions it divides
    if vertical_lines % 2 == 1:
        black_area += (D - B) / 2
    if horizontal_lines % 2 == 1:
        black_area += (C - A) / 2
    if diagonal_lines % 2 == 1:
        black_area += (C - A + D - B) / 4
    
    # Return twice the black area as an integer
    return int(2 * black_area)

import sys
input = sys.stdin.read
data = input().split()
A, B, C, D = map(int, data)
print(calculate_black_area(A, B, C, D))