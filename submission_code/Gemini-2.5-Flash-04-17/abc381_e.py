import sys
import bisect

# Function to solve a single query
def solve_query(L, R, N, ones_prefix, twos_suffix, slash_indices, H_values):
    l_0 = L - 1 # 0-indexed start of substring
    r_0 = R - 1 # 0-indexed end of substring

    # Find the range of slash_indices relevant to the query [l_0, r_0]
    # q_start_idx is the index in slash_indices of the first slash >= l_0
    # q_end_idx is the index in slash_indices of the last slash <= r_0
    q_start_idx = bisect.bisect_left(slash_indices, l_0)
    q_end_idx = bisect.bisect_right(slash_indices, r_0) - 1

    if q_start_idx > q_end_idx:
        # No slashes in the substring
        return 0

    # We want to maximize min(ones_left, twos_right) over slashes in the range.
    # ones_left = ones_prefix[p] - ones_prefix[l_0]
    # twos_right = twos_suffix[p+1] - twos_suffix[r_0 + 1]
    # We maximize min(ones_prefix[p] - ones_prefix[l_0], twos_suffix[p+1] - twos_suffix[r_0 + 1])
    # for p in slash_indices[i] where i in [q_start_idx, q_end_idx]

    # Let C1 = ones_prefix[l_0], C2 = twos_suffix[r_0 + 1]
    # Maximize min(ones_prefix[p] - C1, twos_suffix[p+1] - C2)
    # Let f(i) = ones_prefix[slash_indices[i]] - C1 (non-decreasing in i)
    # Let g(i) = twos_suffix[slash_indices[i]+1] - C2 (non-increasing in i)
    # We want max_{i=q_start_idx}^{q_end_idx} min(f(i), g(i))
    # This max occurs near where f(i) approx g(i), or f(i) - g(i) approx 0.
    # f(i) - g(i) = (ones_prefix[slash_indices[i]] - C1) - (twos_suffix[slash_indices[i]+1] - C2)
    #             = ones_prefix[slash_indices[i]] - twos_suffix[slash_indices[i]+1] + C2 - C1
    #             = H_values[i] + C2 - C1

    # Find the split point using binary search on H_values in the relevant range
    # We search for the first index `split_idx` in the original slash_indices list [q_start_idx, q_end_idx + 1]
    # such that H_values[split_idx] >= ones_prefix[l_0] - twos_suffix[r_0 + 1].
    
    target_H = ones_prefix[l_0] - twos_suffix[r_0 + 1]
    
    # The binary search is effectively on the indices [q_start_idx, q_end_idx] of H_values.
    # bisect_left finds the insertion point `split_idx`.
    # `split_idx` will be an index in the full H_values list, in the range [q_start_idx, q_end_idx + 1].
    # It represents the first index `i` >= q_start_idx where H_values[i] >= target_H.
    split_idx = bisect.bisect_left(H_values, target_H, q_start_idx, q_end_idx + 1)

    max_k = 0

    # Part 1: Consider slashes in the left part [q_start_idx, split_idx - 1] in slash_indices
    # For these indices i, f(i) < g(i), min is f(i). Max f(i) is at the largest i, which is split_idx - 1.
    if split_idx > q_start_idx:
        # The last index in this range in slash_indices is split_idx - 1
        p_left_end = slash_indices[split_idx - 1]
        k_left = ones_prefix[p_left_end] - ones_prefix[l_0]
        max_k = max(max_k, k_left)

    # Part 2: Consider slaches in the right part [split_idx, q_end_idx] in slash_indices
    # For these indices i, f(i) >= g(i), min is g(i). Max g(i) is at the smallest i, which is split_idx.
    if split_idx <= q_end_idx:
         # The first index in this range in slash_indices is split_idx
        p_right_start = slash_indices[split_idx]
        k_right = twos_suffix[p_right_start + 1] - twos_suffix[r_0 + 1]
        max_k = max(max_k, k_right)

    # The maximum possible k is max_k. The length of the 11/22 string is 2*k + 1.
    # If max_k is 0, the length is 1, corresponding to the subsequence '/'.
    # This is valid as long as there is at least one '/' in the substring.
    # The check `if q_start_idx > q_end_idx` handles the case with no slashes (returns 0).
    
    return 2 * max_k + 1

# Read input
# Use sys.stdin.readline for faster input
# sys.stdin = open('input.txt', 'r') # Uncomment for local testing with input.txt
# sys.stdout = open('output.txt', 'w') # Uncomment for local testing with output.txt

N, Q = map(int, sys.stdin.readline().split())
S = sys.stdin.readline().strip()

# Precomputation
# ones_prefix[i] = count of '1' in S[0...i-1] for i in [0, N]
# ones_prefix[0] = 0
ones_prefix = [0] * (N + 1)
for i in range(N):
    ones_prefix[i+1] = ones_prefix[i] + (1 if S[i] == '1' else 0)

# twos_suffix[i] = count of '2' in S[i...N-1] for i in [0, N]
# twos_suffix[N] = 0
twos_suffix = [0] * (N + 1)
for i in range(N - 1, -1, -1):
    twos_suffix[i] = twos_suffix[i+1] + (1 if S[i] == '2' else 0)

# Get indices of '/' (0-indexed)
slash_indices = [i for i, char in enumerate(S) if char == '/']

# Precompute H_values for binary search
# H_values[i] = ones_prefix[slash_indices[i]] - twos_suffix[slash_indices[i]+1]
H_values = [ones_prefix[p] - twos_suffix[p+1] for p in slash_indices]

# Process queries
for _ in range(Q):
    L, R = map(int, sys.stdin.readline().split())
    result = solve_query(L, R, N, ones_prefix, twos_suffix, slash_indices, H_values)
    print(result)