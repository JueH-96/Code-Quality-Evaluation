import math

def find_min_difference(D):
    min_diff = D  # Initialize with the maximum possible difference
    limit = int(math.sqrt(D)) + 1  # We only need to check up to sqrt(D)

    for x in range(limit):
        for y in range(limit):
            current_value = x**2 + y**2
            diff = abs(current_value - D)
            if diff < min_diff:
                min_diff = diff
            if min_diff == 0:
                return 0  # Early exit if we find an exact match

    return min_diff

# Read input
D = int(input().strip())

# Find and print the minimum difference
print(find_min_difference(D))