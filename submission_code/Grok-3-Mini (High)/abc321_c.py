import math
import itertools

# Read K from input
K = int(input())

# Compute number of 321-like numbers for each digit length
num_with_d = [0] * 11
num_with_d[1] = 9  # For digit length 1, only 1-9
for d in range(2, 11):
    num_with_d[d] = math.comb(10, d)

# Compute cumulative sum up to each digit length
cum_up_to = [0] * 11
for d in range(1, 11):
    cum_up_to[d] = cum_up_to[d - 1] + num_with_d[d]

# Find the digit length d where the K-th number falls
for d in range(1, 11):
    if cum_up_to[d] >= K:
        break

# Cumulative sum up to digit length d-1
prev_cum = cum_up_to[d - 1]

# Rank of the number within the d-digit 321-like numbers (1-based)
rank = K - prev_cum

# Generate combinations based on digit length
if d == 1:
    # For digit length 1, combinations from 1 to 9
    combs = list(itertools.combinations(range(1, 10), d))
else:
    # For digit length >= 2, combinations from 0 to 9
    combs = list(itertools.combinations(range(10), d))

# Create descending order tuples
desc_combs = [tuple(reversed(comb)) for comb in combs]

# Sort the descending tuples in ascending lexicographical order
sorted_desc_combs = sorted(desc_combs)

# Get the digits of the rank-th number (0-based index)
digits = sorted_desc_combs[rank - 1]

# Form the number from the digits
num_str = ''.join(map(str, digits))
number = int(num_str)

# Output the number
print(number)