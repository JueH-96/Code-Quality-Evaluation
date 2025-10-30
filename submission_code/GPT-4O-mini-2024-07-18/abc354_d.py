def calculate_black_area(A, B, C, D):
    # Count the number of black squares in the rectangle defined by (A, B) and (C, D)
    black_count = 0
    
    # Iterate through the rectangle defined by A, B, C, D
    for x in range(A, C):
        for y in range(B, D):
            # Determine the color of the region at (x, y)
            # The region is black if the sum of the coordinates (x + y) is even
            # and if (x, y) is in the correct parity region
            if (x % 2 + y % 2) % 2 == 0:
                black_count += 1

    # The area of black regions is counted, we need to return twice that area
    return black_count * 2

# Read input
import sys
input = sys.stdin.read
A, B, C, D = map(int, input().strip().split())

# Calculate and print the result
result = calculate_black_area(A, B, C, D)
print(result)