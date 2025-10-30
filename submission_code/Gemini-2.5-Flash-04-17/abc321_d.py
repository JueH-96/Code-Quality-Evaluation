import sys
from bisect import bisect_left

# Read input
# Using map(int, sys.stdin.readline().split()) is efficient for reading integers
N, M, P = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))

# Sort B to efficiently find elements less than a threshold using binary search
B.sort()

# Compute prefix sums of sorted B
# prefix_sum_B[i] stores the sum of the first i elements of the sorted B array.
# prefix_sum_B has size M+1, with prefix_sum_B[0] = 0.
# prefix_sum_B[i] = B[0] + ... + B[i-1] for i > 0.
# Use Python's arbitrary precision int to avoid overflow
prefix_sum_B = [0] * (M + 1)
for i in range(M):
    prefix_sum_B[i+1] = prefix_sum_B[i] + B[i]

# Calculate total price of all set meals
# Use Python's arbitrary precision int for the total sum
total_price = 0

# Iterate through each main dish price 'a'
for a in A:
    # For a fixed main dish price 'a', we consider all side dish prices 'b' from B.
    # The price of a set meal (a, b) is min(a + b, P).
    # We can split the side dishes 'b' into two groups based on the condition a + b < P:
    # Group 1: b such that a + b < P (or b < P - a). For these, price is a + b.
    # Group 2: b such that a + b >= P (or b >= P - a). For these, price is P.

    # Find the threshold value for b
    # A side dish 'b' belongs to Group 1 if b < P - a.
    limit = P - a
    
    # Use binary search on sorted B to find the split point.
    # bisect_left(B, limit) returns the index k such that all elements B[0...k-1] are < limit,
    # and all elements B[k...M-1] are >= limit.
    # So, k is the count of side dishes in Group 1.
    # Note: bisect_left handles cases where limit is smaller than all elements (k=0)
    # or larger than all elements (k=M).
    k = bisect_left(B, limit)

    # Calculate the sum of prices for side dishes in Group 1 (k elements)
    # For each b_j = B[j] where j < k, price is a + b_j.
    # Sum = sum_{j=0 to k-1} (a + B[j]) = sum_{j=0 to k-1} a + sum_{j=0 to k-1} B[j]
    # Sum = k * a + prefix_sum_B[k]
    sum_price_less_than_P = k * a + prefix_sum_B[k]

    # Calculate the sum of prices for side dishes in Group 2 (M - k elements)
    # For these M-k elements B[j] where j >= k, price is P.
    # Sum = (M - k) * P
    sum_price_greater_equal_P = (M - k) * P

    # The total price for the current main dish 'a' combined with all side dishes
    total_price_for_a = sum_price_less_than_P + sum_price_greater_equal_P

    # Add this to the overall total price
    total_price += total_price_for_a

# Print the final total price
print(total_price)