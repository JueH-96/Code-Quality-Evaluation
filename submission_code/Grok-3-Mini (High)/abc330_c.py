import sys
import math

# Read input from stdin
data = sys.stdin.readline().strip()
D = int(data)

# Compute the maximum x value, which is floor(sqrt(D))
max_x = int(math.sqrt(D))

# Initialize minimum difference to D, which is an upper bound
min_diff = D

# Iterate over all possible x from 0 to max_x inclusive
for x in range(0, max_x + 1):
    # Compute the remainder rem = D - x^2
    rem = D - x * x
    # Compute s = floor(sqrt(rem))
    s = int(math.sqrt(rem))
    # Compute the two possible differences
    diff1 = rem - s * s  # rem - s^2
    diff2 = (s + 1) * (s + 1) - rem  # (s+1)^2 - rem
    # Find the minimum difference for this x
    current_min = min(diff1, diff2)
    # Update the overall minimum difference
    if current_min < min_diff:
        min_diff = current_min

# Output the minimum difference
print(min_diff)