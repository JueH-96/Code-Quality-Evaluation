# YOUR CODE HERE
# Read the integer N from standard input
N = int(input())

# Calculate the remainder when N is divided by 5
remainder = N % 5

# Determine the nearest multiple of 5 based on the remainder
# Water stations are at positions 0, 5, 10, ..., 100. These are multiples of 5.
# We want to find the multiple of 5 closest to N.
# The remainder `r = N % 5` tells us how far N is from the previous multiple of 5: `N = (N // 5) * 5 + r`.
# The previous multiple is `N - r`. The next multiple is `N + (5 - r)`.
# We compare the distance to the previous multiple `r` with the distance to the next multiple `5 - r`.
# If `r <= 5 - r`, which simplifies to `2 * r <= 5` or `r <= 2.5`, the previous multiple is nearer or equally near.
# Since `r` is an integer (0, 1, 2, 3, or 4), `r <= 2.5` means `r` is 0, 1, or 2.
# If `r <= 2`, the nearest station is the previous multiple: `N - remainder`.
# If `r > 2` (i.e., r is 3 or 4), the nearest station is the next multiple: `N + (5 - remainder)`.
if remainder <= 2:
    nearest_station = N - remainder
else: # remainder is 3 or 4
    nearest_station = N + (5 - remainder)

# Print the result to standard output
print(nearest_station)