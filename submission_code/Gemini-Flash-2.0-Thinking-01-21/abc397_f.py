import sys

# Use faster input reading
input = sys.stdin.readline

# Increase recursion depth if needed (Segment Tree can be recursive)
# Python default recursion depth is usually 1000.
# For N=3e5, the segment tree depth is log2(3e5) ~ 18-19. Standard recursion depth is usually sufficient.
# sys.setrecursionlimit(3000)

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    # 0-based indexing for A: A[0...N-1]
    # Splits: A[0:i], A[i:j], A[j:N]
    # Valid i: 1 <= i <= N-2 (first split point is index i)
    # Valid j: i+1 <= j <= N-1 (second split point is index j)

    # Precompute distinct count for prefixes A[0:k] for k = 1...N
    # distinct_prefix[k] = count distinct in A[0:k] (length k)
    distinct_prefix = [0] * (N + 1)
    freq = {}
    count = 0
    for k in range(N):
        freq[A[k]] = freq.get(A[k], 0) + 1
        if freq[A[k]] == 1:
            count += 1
        distinct_prefix[k+1] = count

    # Precompute distinct count for suffixes A[k:N] for k = 0...N-1
    # distinct_suffix[k] = count distinct in A[k:N] (length N-k)
    distinct_suffix = [0] * (N + 1)
    freq = {}
    count = 0
    for k in range(N - 1, -1, -1):
        freq[A[k]] = freq.get(A[k], 0) + 1
        if freq[A[k]] == 1:
            count += 1
        distinct_suffix[k] = count

    # Precompute previous occurrence index for each element
    # prev_occurrence[k] = index of previous A[k] before k, or -1 if none
    prev_occurrence = [-1] * N
    last_occurrence = {} # Stores the last seen index of each value
    for k in range(N):
        if A[k] in last_occurrence:
            prev_occurrence[k] = last_occurrence[A[k]]
        last_occurrence[A[k]] = k

    # Segment tree on first split indices i from 1 to N-2.
    # Segment tree size N-2, indices 0 to N-3.
    # Segment tree index `idx` corresponds to first split index `i = idx + 1`.
    seg_tree_size = N - 2

    # Initialize segment tree with a value smaller than any possible result
    # Minimum possible distinct count in a non-empty segment is 1.
    # Minimum total sum is 1 + 1 + 1 = 3.
    # Use a sufficiently small number like -10^9.
    initial_tree_val = -10**9

    tree = [initial_tree_val] * (4 * seg_tree_size)
    lazy = [0] * (4 * seg_tree_size)

    def push(v, tl, tr):
        if lazy[v] != 0:
            tree[v] += lazy[v]
            if tl != tr:
                lazy[v*2] += lazy[v]
                lazy[v*2+1] += lazy[v]
            lazy[v] = 0

    def update_range(v, tl, tr, l, r, add):
        push(v, tl, tr)
        if l > r or tl > r or tr < l:
            return
        if l <= tl and tr <= r:
            lazy[v] += add
            push(v, tl, tr)
        else:
            tm = (tl + tr) // 2
            update_range(v*2, tl, tm, l, r, add)
            update_range(v*2+1, tm+1, tr, l, r, add)
            tree[v] = max(tree[v*2], tree[v*2+1])

    def query_max(v, tl, tr, l, r):
        if l > r or tl > r or tr < l:
            return initial_tree_val # Return identity for max (very small number)
        push(v, tl, tr)
        if l <= tl and tr <= r:
            return tree[v]
        tm = (tl + tr) // 2
        return max(query_max(v*2, tl, tm, l, r),
                   query_max(v*2+1, tm+1, tr, l, r))

    # Build segtree for i in [1, N-2] (indices 0 to N-3).
    # Initial value at segtree index `i-1` is `distinct_prefix[i]`.
    # This represents `distinct_prefix[i] + count_distinct(A[i:i])` where count_distinct(A[i:i]) is 0.
    def build_init(v, tl, tr):
        if tl == tr:
            # Index `tl` in segtree corresponds to first split `i = tl + 1`.
            # Set initial value to distinct_prefix[i].
            tree[v] = distinct_prefix[tl + 1]
        else:
            tm = (tl + tr) // 2
            build_init(v*2, tl, tm)
            build_init(v*2+1, tm+1, tr)
            tree[v] = max(tree[v*2], tree[v*2+1])

    # Constraint N >= 3 ensures seg_tree_size = N - 2 >= 1.
    build_init(1, 0, seg_tree_size - 1)

    max_total_sum = 0

    # Iterate through possible second split points j (0-based index)
    # j ranges from 2 to N-1
    for j in range(2, N):
        # Process the effect of adding A[j-1] to segments A[i:j-1] for i < j.
        # The term `count_distinct(A[i:j])` increases by 1 compared to `count_distinct(A[i:j-1])`
        # if A[j-1] is not in A[i:j-1]. This happens if `prev_occurrence[j-1] < i`.
        # p = index of previous occurrence of A[j-1] before j-1 (0-based index in A)
        p = prev_occurrence[j-1]

        # The indices `i` (1-based split index) that get +1 are where `p < i <= j-1`.
        # We only care about `i` in `[1, N-2]` (the domain of the segment tree).
        # The update range for `i` is `[\max(1, p + 1), \min(j - 1, N - 2)]`.
        # Convert `i` range to segtree index range `[idx, idx]`: `[\max(1, p + 1) - 1, \min(j - 1, N - 2) - 1]`.
        # `\max(1, p+1) - 1` maps to `max(0, p)` for segtree index (if p is -1, 0, 1...).
        # `\min(j-1, N-2) - 1` maps to `min(j-2, N-3)` for segtree index.
        update_l_seg = max(0, p)
        update_r_seg = min(seg_tree_size - 1, j - 2)

        if update_l_seg <= update_r_seg:
             update_range(1, 0, seg_tree_size - 1, update_l_seg, update_r_seg, 1)

        # After the update, the segment tree stores `distinct_prefix[i] + count_distinct(A[i:j])`
        # at index `i-1` for `i \in [1, \min(j-1, N-2)]`.

        # We need the maximum of `distinct_prefix[i] + count_distinct(A[i:j])`
        # over possible first split points `i` for the current `j`.
        # The possible `i` values are `1` to `j-1`.
        # We only consider `i` up to `N-2`. So, the range for `i` is `[1, \min(j-1, N-2)]`.
        # Convert `i` range to segtree index range `[idx, idx]`: `[1 - 1, \min(j - 1, N - 2) - 1] = [0, \min(j - 2, N - 3)]`.
        query_l_seg = 0
        query_r_seg = min(seg_tree_size - 1, j - 2)

        if query_l_seg <= query_r_seg:
            # Query the maximum value in the segment tree over the valid range of `i`.
            max_prefix_mid_j = query_max(1, 0, seg_tree_size - 1, query_l_seg, query_r_seg)
            # The total sum for this second split at `j` is `max_prefix_mid_j + distinct_suffix[j]`.
            # `distinct_suffix[j]` is for A[j:N].
            max_total_sum = max(max_total_sum, max_prefix_mid_j + distinct_suffix[j])

    print(max_total_sum)

solve()