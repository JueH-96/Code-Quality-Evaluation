import sys

# Read input
# N: length of sequence A
# Q: number of queries
N, Q = map(int, sys.stdin.readline().split())
# queries: list of integers x_i for each query
queries = list(map(int, sys.stdin.readline().split()))

# 1. Simulate the set S for each query and record its size.
# S will store elements that are currently in the set.
S = set()
# sizes_list[i] will store |S_{i+1}|, the size of set S after processing the (i+1)-th query (0-indexed list).
# So, sizes_list[i-1] corresponds to |S_i| using 1-based query index i.
sizes_list = []
for x in queries:
    if x in S:
        S.remove(x)
    else:
        S.add(x)
    sizes_list.append(len(S))

# 2. Compute prefix sums of sizes_list.
# prefix_sizes[k] will store the sum of the first k elements of sizes_list.
# prefix_sizes[k] = sum(sizes_list[0] ... sizes_list[k-1]) = sum_{i=1}^k |S_i| (using 1-based query index i)
# prefix_sizes[0] = 0
prefix_sizes = [0]
current_sum = 0
for size in sizes_list:
    current_sum += size
    prefix_sizes.append(current_sum)
# prefix_sizes has length Q+1. prefix_sizes[k] corresponds to sum up to query k (1-based).

# 3. Group query indices by element value.
# toggle_times[j] stores the 1-based query indices i where x_i = j.
# This tells us when element j is added/removed from the set S.
# The list at index 0 is unused as element values are 1..N.
toggle_times = [[] for _ in range(N + 1)]
for i in range(Q):
    element = queries[i] # queries list is 0-indexed
    # Store the 1-based query index (i + 1)
    toggle_times[element].append(i + 1)

# 4. Initialize the sequence A.
# A[j-1] corresponds to A_j. Initially all elements are 0.
A = [0] * N

# 5. Compute A_j for each element j from 1 to N.
# A_j is the sum of |S_i| for all queries i where element j was in S after the update.
# Element j is in S during the query intervals determined by its toggle times.
for j in range(1, N + 1):
    times = toggle_times[j]
    k_j = len(times) # Number of times element j appeared in queries

    # The list 'times' contains the 1-based query indices where j was toggled.
    # Since j starts outside S, it enters S at times[0], leaves at times[1], enters at times[2], etc.
    # j is in S during the intervals:
    # [times[0], times[1]-1], [times[2], times[3]-1], ..., [times[2p], times[2p+1]-1], ...
    # If k_j is odd, the last interval is [times[k_j-1], Q] (j stays in S until the end).

    # Iterate through the toggle times, taking pairs as start/end markers.
    # The step is 2 because times[p] is the start of an 'in S' interval and times[p+1] is the start of an 'out of S' interval.
    for p in range(0, k_j, 2):
        # The start query index (1-based) of the interval where j is in S.
        start_query_idx = times[p]

        # The end query index (1-based) of the interval where j is in S.
        # This is either the query *before* the next toggle time (if one exists),
        # or the last query Q (if this is the last toggle time for j).
        if p + 1 < k_j:
            # There is a next toggle time times[p+1]. j is removed at times[p+1], so it was in S up to times[p+1] - 1.
            end_query_idx = times[p+1] - 1
        else:
            # This is the last toggle time for j. j is added/re-added at times[p] and stays in S until the end of all queries.
            end_query_idx = Q

        # The interval of queries where j is in S is [start_query_idx, end_query_idx] (1-based inclusive).
        # We need to sum |S_i| for i ranging from start_query_idx to end_query_idx.
        # Using the prefix sum array: sum(|S_i| for i=L to R) = prefix_sizes[R] - prefix_sizes[L-1].
        # Note: The prefix_sizes array is 1-indexed conceptually with respect to queries.
        # prefix_sizes[k] stores sum of |S_1|...|S_k|.
        # The indices L-1 and R correspond to prefix_sizes lookups.

        # The condition start_query_idx <= end_query_idx holds because the query indices in 'times' are strictly increasing.
        # If start_query_idx > end_query_idx (e.g., interval [5, 4]), the prefix sum difference will be 0, which is correct.
        
        A[j - 1] += prefix_sizes[end_query_idx] - prefix_sizes[start_query_idx - 1]

# 6. Print the final sequence A.
# Print elements space-separated.
print(*A)