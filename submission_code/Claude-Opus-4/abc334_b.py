# YOUR CODE HERE
import math

A, M, L, R = map(int, input().split())

# Find the range of k values
# L <= A + kM <= R
# (L - A) / M <= k <= (R - A) / M

# Calculate the minimum k (ceiling of (L-A)/M)
if L - A >= 0:
    k_min = (L - A + M - 1) // M
else:
    k_min = (L - A) // M

# Calculate the maximum k (floor of (R-A)/M)
if R - A >= 0:
    k_max = (R - A) // M
else:
    k_max = (R - A - M + 1) // M

# Count the number of valid k values
if k_min <= k_max:
    count = k_max - k_min + 1
else:
    count = 0

print(count)