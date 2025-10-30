N = int(input())

repunits = []
current_r = 0
# Based on analysis, repunits up to R_12 (12 ones) are sufficient.
# The number of distinct sums from R_1...R_12 is C(12+3-1, 3) = C(14, 3) = 364.
# N is at most 333, so this range is enough.
# Also, any sum involving R_13 or higher is larger than any sum using only R_1...R_12.
MAX_REPUNIT_LEN = 12

for _ in range(MAX_REPUNIT_LEN):
    current_r = current_r * 10 + 1
    repunits.append(current_r)

sums_list = []
# Generate all unique sums of three repunits
# by choosing repunits with indices i, j, k such that i <= j <= k.
# This ensures each combination of repunits is generated exactly once.
# Example: repunits[0] is R_1, repunits[1] is R_2, ..., repunits[11] is R_12.
for i in range(MAX_REPUNIT_LEN):
    for j in range(i, MAX_REPUNIT_LEN):
        for k in range(j, MAX_REPUNIT_LEN):
            sum_val = repunits[i] + repunits[j] + repunits[k]
            sums_list.append(sum_val)

# Sort the list of sums in ascending order
sums_list.sort()

# Print the N-th smallest sum. N is 1-indexed, so access list at N-1.
print(sums_list[N-1])