import math

def get_black_area(A, B, C, D):
    black_area = 0

    # Iterate over the x-coordinates
    for x in range(A, C+1):
        # Iterate over the y-coordinates
        for y in range(B, D+1):
            # Check if the current point is black
            if (x + y) % 2 == 0 and (x % 2 == 0 or y % 2 == 0):
                black_area += 1

    return black_area * 2

# Read input from stdin
A, B, C, D = map(int, input().split())

# Compute and print the answer
print(get_black_area(A, B, C, D))