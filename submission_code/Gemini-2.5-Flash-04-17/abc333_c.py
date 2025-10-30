# YOUR CODE HERE
import sys

# Read the input N
N = int(sys.stdin.readline())

# Generate repunits
# A repunit is an integer whose digits are all 1.
# We need to find the N-th smallest sum of exactly three repunits.
# Let R_i be the repunit with i digits (i.e., i ones). R_1=1, R_2=11, R_3=111, ...
# We are considering sums R_a + R_b + R_c where a, b, c >= 1.
# The problem does not require a, b, c to be distinct.
# To find the N-th smallest sum, we can generate all possible sums and sort them.
# We need to determine how many repunits to consider.
# Let's consider repunits R_1, R_2, ..., R_M. The sums are R_i + R_j + R_k where 1 <= i, j, k <= M.
# The number of unique sums of the form R_a + R_b + R_c with 1 <= a <= b <= c <= M is (M+3-1 choose 3) = (M+2 choose 3).
# We need this number to be at least N to guarantee that the N-th smallest sum is included.
# Given the constraint N <= 333:
# If M=11, (11+2 choose 3) = (13 choose 3) = (13 * 12 * 11) // (3 * 2 * 1) = 13 * 2 * 11 = 286 (not enough for N=333).
# If M=12, (12+2 choose 3) = (14 choose 3) = (14 * 13 * 12) // (3 * 2 * 1) = 14 * 13 * 2 = 364 (enough for N=333).
# Thus, we need to consider the first M=12 repunits (R_1 to R_12).

M = 12 # Number of repunits to generate
rep = []
current_rep = 0
for _ in range(M):
    current_rep = current_rep * 10 + 1
    rep.append(current_rep)

# Generate all possible sums of three repunits from the list 'rep'.
# The list 'rep' contains R_1, R_2, ..., R_M at indices 0, 1, ..., M-1 respectively.
# So, rep[i] corresponds to R_{i+1}.
# We generate sums rep[i] + rep[j] + rep[k] for 0 <= i, j, k < M.
# To get unique sums efficiently and ensure we cover all combinations with replacement,
# we iterate through indices such that 0 <= i <= j <= k < M.
# As shown in analysis, the sum rep[i] + rep[j] + rep[k] is unique for each unique tuple (i, j, k)
# satisfying 0 <= i <= j <= k < M.
sums_list = []
for i in range(M):
    for j in range(i, M): # Ensure the index j is not less than i
        for k in range(j, M): # Ensure the index k is not less than j
            sums_list.append(rep[i] + rep[j] + rep[k])

# The list sums_list now contains all unique sums R_a + R_b + R_c with 1 <= a <= b <= c <= 12.
# The number of elements in sums_list is exactly (12+2 choose 3) = 364.

# Sort the list of sums in ascending order
sums_list.sort()

# The N-th smallest sum is at index N-1 in the sorted list (since lists are 0-indexed).
# N is between 1 and 333, so N-1 is between 0 and 332.
# The list sums_list has 364 elements (indices 0 to 363), so accessing sums_list[N-1] is always valid.
print(sums_list[N-1])