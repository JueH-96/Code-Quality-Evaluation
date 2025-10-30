import math

# Read input
R = int(input())

# Calculate R^2
R_sq = R * R

# The condition for a square centered at (i, j) to be completely inside the circle
# is that its farthest corner is within distance R from the origin (0, 0).
# The corners are (i +/- 0.5, j +/- 0.5). The farthest corner is at (|i|+0.5, |j|+0.5) in terms of absolute coordinates.
# The condition is (|i|+0.5)^2 + (|j|+0.5)^2 <= R^2.

# We need to count the number of integer pairs (i, j) satisfying this condition.
# The condition depends only on |i| and |j|.
# For a fixed integer i, we count the number of integers j such that the condition holds.
# (|j|+0.5)^2 <= R^2 - (|i|+0.5)^2
# If R^2 - (|i|+0.5)^2 < 0, there are no solutions for |j| >= 0.
# Otherwise, |j|+0.5 <= sqrt(R^2 - (|i|+0.5)^2).
# |j| <= sqrt(R^2 - (|i|+0.5)^2) - 0.5.
# Let K = floor(sqrt(R^2 - (|i|+0.5)^2) - 0.5).
# If K < 0, there are no non-negative integer solutions for |j|, so no integer solutions for j. Count is 0.
# If K >= 0, the possible integer values for |j| are 0, 1, ..., K.
# If |j|=0, j=0 (1 value).
# If |j|=k for k in {1, ..., K}, j=k or j=-k (2 values).
# Total number of integer j values for a fixed i is 1 + 2 * K, provided K >= 0. If K < 0, count is 0.

# We need to sum this count over all integers i.
# The possible range of |i| for K >= 0 is when sqrt(R^2 - (|i|+0.5)^2) - 0.5 >= 0.
# sqrt(R^2 - (|i|+0.5)^2) >= 0.5
# R^2 - (|i|+0.5)^2 >= 0.25
# (|i|+0.5)^2 <= R^2 - 0.25
# |i|+0.5 <= sqrt(R^2 - 0.25)  (Since |i|+0.5 >= 0.5 > 0)
# |i| <= sqrt(R^2 - 0.25) - 0.5.
# The maximum integer value for |i| is floor(sqrt(R^2 - 0.25) - 0.5).
# Let M = floor(sqrt(R_sq - 0.25) - 0.5).
# The possible integer values for i are -M, -M+1, ..., 0, ..., M-1, M.

# Total count = sum_{i=-M}^{M} (1 + 2 * floor(sqrt(R_sq - (|i|+0.5)^2) - 0.5))
# Note: For R >= 1, R_sq >= 1, R_sq - 0.25 >= 0.75. sqrt(...) >= sqrt(0.75) approx 0.866. sqrt(...) - 0.5 >= 0.366. Floor >= 0.
# So M = floor(sqrt(R_sq - 0.25) - 0.5) is always >= 0 for R >= 1.
# Also, for any |i| <= M, (|i|+0.5)^2 <= (M+0.5)^2 <= (sqrt(R_sq - 0.25))^2 = R_sq - 0.25.
# So R_sq - (|i|+0.5)^2 >= 0.25 >= 0.
# This implies sqrt(R_sq - (|i|+0.5)^2) >= 0.5.
# And sqrt(R_sq - (|i|+0.5)^2) - 0.5 >= 0.
# So floor(sqrt(R_sq - (|i|+0.5)^2) - 0.5) is always >= 0 for |i| <= M.
# The max(0, ...) is not strictly needed within the summation range [-M, M].

# Summation:
# Term for i=0: 1 + 2 * floor(sqrt(R_sq - (0.5)^2) - 0.5)
# Terms for i > 0: sum_{i=1}^{M} (1 + 2 * floor(sqrt(R_sq - (i+0.5)^2) - 0.5))
# Terms for i < 0: sum_{i=-M}^{-1} (1 + 2 * floor(sqrt(R_sq - (|-i|+0.5)^2) - 0.5))
# Since |i| = |-i|, the terms for i and -i are the same when i != 0.
# sum_{i=-M}^{-1} term(i) = sum_{i'=1}^{M} term(i') where i' = -i.
# Total count = term(i=0) + 2 * sum_{i=1}^{M} term(i).

total_count = 0

# Calculate M, the maximum absolute integer value for i.
# This is the upper limit for the loop iterating through |i|.
# M = floor(sqrt(R_sq - 0.25) - 0.5)
# For R >= 1, R_sq >= 1, R_sq - 0.25 >= 0.75 > 0. So sqrt(R_sq - 0.25) is real and positive.
M = math.floor(math.sqrt(R_sq - 0.25) - 0.5)

# Handle i = 0
# K_0 = floor(sqrt(R_sq - (0.5)^2) - 0.5) = floor(sqrt(R_sq - 0.25) - 0.5)
# We already established that for R >= 1, K_0 >= 0.
K_0 = math.floor(math.sqrt(R_sq - 0.25) - 0.5)
total_count += (1 + 2 * K_0)

# Handle i = 1, 2, ..., M
# For each such i, we add the count for +i and -i.
# The sum limit for i should go up to M.
# For i in range(1, int(M) + 1), the term is (1 + 2 * floor(sqrt(R_sq - (i+0.5)^2) - 0.5)).
# We already established that for i in [1, M], floor(sqrt(R_sq - (i+0.5)^2) - 0.5) >= 0.
for i in range(1, int(M) + 1):
    # Calculate K_i = floor(sqrt(R_sq - (i+0.5)^2) - 0.5)
    val_i = R_sq - (i + 0.5)**2
    # We know val_i >= 0 for i <= M when R >= 1, specifically val_i >= 0.25.
    K_i = math.floor(math.sqrt(val_i) - 0.5)
    
    # The number of j values for fixed i is (1 + 2 * K_i).
    # Since we are summing for i = 1 to M, these represent i and -i pairs.
    # Total count += 2 * (number of j values for this i)
    total_count += 2 * (1 + 2 * K_i)

# Print the result
print(total_count)