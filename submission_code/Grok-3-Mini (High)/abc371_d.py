import sys
import bisect

# Read all input from stdin
data = sys.stdin.read().split()
index = 0

# Read N
N = int(data[index])
index += 1

# Read X coordinates
X = list(map(int, data[index:index + N]))
index += N

# Read P populations
P = list(map(int, data[index:index + N]))
index += N

# Read Q
Q = int(data[index])
index += 1

# Compute prefix sum of populations
prefix_sum = [0]
cum_sum = 0
for p in P:
    cum_sum += p
    prefix_sum.append(cum_sum)

# Process each query
for _ in range(Q):
    L_val = int(data[index])
    R_val = int(data[index + 1])
    index += 2
    
    # Find the start index using binary search (first position >= L_val)
    start_idx = bisect.bisect_left(X, L_val)
    
    # Find the end index using binary search (first position > R_val, then subtract 1)
    right_insert = bisect.bisect_right(X, R_val)
    end_idx = right_insert - 1
    
    # Check if there are villages in the range [L_val, R_val]
    if 0 <= start_idx <= end_idx < N:
        # Calculate sum using prefix sum
        sum_pop = prefix_sum[end_idx + 1] - prefix_sum[start_idx]
        print(sum_pop)
    else:
        # No villages in range
        print(0)