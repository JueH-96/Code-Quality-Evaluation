import sys
import bisect

MOD = 998244353

data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))
index += N

# Compute all possible differences
diffs = set()
for i in range(N):
    for j in range(i + 1, N):
        diff = A[j] - A[i]
        diffs.add(diff)

# Group indices by value
val_to_indices = {}
for idx in range(N):
    val = A[idx]
    if val not in val_to_indices:
        val_to_indices[val] = []
    val_to_indices[val].append(idx)  # list is sorted as indices are added in order

# Initialize total arithmetic sum for each length k=1 to N
total_arith_sum = [0] * N

# For each difference, compute the DP and add to total
for d in diffs:
    # Initialize prev_dp for L=0, all 1
    prev_dp = [1] * N
    # For each L from 1 to N-1
    for L in range(1, N):
        # Compute prefix sum dict for current prev_dp
        prefix_sum_dict = {}
        for w_val, idx_list in val_to_indices.items():
            dp_values = [prev_dp[idx] for idx in idx_list]
            cum_sum = [0]
            accum = 0
            for val in dp_values:
                accum = (accum + val) % MOD
                cum_sum.append(accum)
            prefix_sum_dict[w_val] = cum_sum
        # Now compute current_dp
        current_dp = [0] * N
        for i in range(N):
            w_target = A[i] - d  # value for predecessors
            if w_target in val_to_indices:
                idx_list = val_to_indices[w_target]
                cum_sum = prefix_sum_dict[w_target]
                k = bisect.bisect_left(idx_list, i)  # number of u < i in idx_list
                sum_val = cum_sum[k]
            else:
                sum_val = 0
            current_dp[i] = sum_val
        # Compute sum_curr for length k = L + 1
        sum_curr = sum(current_dp) % MOD
        # Add to total_arith_sum for index L (length k = L + 1)
        total_arith_sum[L] = (total_arith_sum[L] + sum_curr) % MOD
        # Set prev_dp to current_dp for next L
        prev_dp = list(current_dp)  # make a copy

# Set the value for k=1
total_arith_sum[0] = N

# Output the results
print(' '.join(map(str, total_arith_sum)))