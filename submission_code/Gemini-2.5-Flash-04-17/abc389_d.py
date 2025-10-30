import sys
import math

R = int(sys.stdin.readline())

total_count = 0
R_sq_float = float(R) * R

# Calculate the maximum integer i >= 0 such that there is at least one integer j
# satisfying (i+0.5)^2 + (|j|+0.5)^2 <= R^2.
# The smallest possible value for (|j|+0.5)^2 for an integer |j| >= 0 is 0.25 (when |j|=0).
# So, for a given i, a non-negative integer |j| is possible only if
# (i+0.5)^2 + 0.25 <= R^2
# (i+0.5)^2 <= R^2 - 0.25
# Taking the square root (both sides are non-negative since R >= 1):
# i+0.5 <= sqrt(R^2 - 0.25)
# i <= sqrt(R^2 - 0.25) - 0.5
# The maximum integer i >= 0 satisfying this inequality is floor(sqrt(R^2 - 0.25) - 0.5).

# For R >= 1, R^2 >= 1, so R^2 - 0.25 >= 0.75 > 0. The square root is well-defined and positive.
# sqrt(0.75) is approximately 0.866, so sqrt(R^2 - 0.25) - 0.5 >= 0.866 - 0.5 = 0.366.
# Thus, floor(sqrt(R^2 - 0.25) - 0.5) will be >= 0 for R >= 1.
# So max_i will be >= 0.

max_i = int(math.sqrt(R_sq_float - 0.25) - 0.5)

# Loop through possible integer values of i from 0 up to max_i
# For each such i, find the number of integer values of j.
for i in range(max_i + 1):
    # For this fixed i, we need to find integer j such that (|j|+0.5)^2 <= R^2 - (i+0.5)^2.
    # Let val_j_sq_float = R^2 - (i+0.5)^2.
    # Based on how max_i was calculated, for i <= max_i, we know that
    # (i+0.5)^2 <= R^2 - 0.25, which implies R^2 - (i+0.5)^2 >= 0.25.
    # So val_j_sq_float is guaranteed to be >= 0.25 (within floating-point precision).

    i_plus_0_5 = i + 0.5
    i_plus_0_5_sq = i_plus_0_5 * i_plus_0_5
    val_j_sq_float = R_sq_float - i_plus_0_5_sq

    # We need (|j| + 0.5)^2 <= val_j_sq_float.
    # Taking the square root (both sides are non-negative since val_j_sq_float >= 0.25):
    # |j| + 0.5 <= sqrt(val_j_sq_float)
    # |j| <= sqrt(val_j_sq_float) - 0.5
    # The maximum integer value for |j| is Y_max = floor(sqrt(val_j_sq_float) - 0.5).
    # Since val_j_sq_float >= 0.25, sqrt(val_j_sq_float) >= 0.5, so sqrt(...) - 0.5 >= 0.
    # Thus, Y_max will be an integer >= 0.
    Y_max = int(math.sqrt(val_j_sq_float) - 0.5)

    # The possible integer values for |j| are 0, 1, ..., Y_max.
    # There are Y_max + 1 such non-negative |j| values.

    # Now, count the corresponding integer pairs (i, j).
    if i == 0:
        # For i=0:
        # The value |j|=0 corresponds to j=0. This gives the pair (0,0). (1 pair)
        # Each value |j|=k > 0 (for k = 1, ..., Y_max) corresponds to j=k and j=-k. (2 pairs each)
        # Number of |j|=k > 0 cases is Y_max.
        # Total pairs for i=0: 1 * (for |j|=0) + 2 * Y_max (for |j|>0)
        total_count += (1 + 2 * Y_max)
    else: # i > 0
        # For i > 0:
        # The index i corresponds to two x-coordinates: i and -i.
        # For a fixed i > 0 and a fixed non-negative |j|:
        # If |j|=0 (j=0): Pairs are (i, 0) and (-i, 0). (2 pairs)
        # If |j|=k > 0 (for k = 1, ..., Y_max): Pairs are (i, k), (i, -k), (-i, k), (-i, -k). (4 pairs)
        # Number of |j|=0 cases: 1 (j=0).
        # Number of |j|=k > 0 cases: Y_max.
        # Total pairs for this i > 0 (including both i and -i):
        # 2 * (number of |j|=0 cases) + 4 * (number of |j|>0 cases)
        # 2 * 1 + 4 * Y_max
        total_count += 2 * (1 + 2 * Y_max)

print(total_count)