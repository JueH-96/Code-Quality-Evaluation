import sys

# Segment tree with lazy propagation for range add and range max query
# Tree indices are 0-based, representing values from original problem `i` as `i-1`.
class SegmentTree:
    def __init__(self, size, initial_values):
        self.size = size
        # If size is 0, it means there are no valid 'i' values to build a segment tree for.
        # This occurs if N-2 <= 0, i.e., N <= 2.
        # Problem constraint N >= 3, so size >= 1 always. This check is not strictly needed but harmless.
        if size == 0: 
            return 
        self.tree = [0] * (4 * size) # Stores maximum value in range
        self.lazy = [0] * (4 * size) # Stores pending additions for a range
        self._build(initial_values, 1, 0, size - 1)

    def _build(self, initial_values, node, start, end):
        # Base case: a leaf node
        if start == end:
            self.tree[node] = initial_values[start]
        else:
            mid = (start + end) // 2
            # Recursively build left and right children
            self._build(initial_values, 2 * node, start, mid)
            self._build(initial_values, 2 * node + 1, mid + 1, end)
            # Current node's value is the max of its children
            self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])

    def _push(self, node):
        # Push lazy updates to children if current node has pending updates
        if self.lazy[node] != 0:
            # Apply lazy value to children's tree values
            self.tree[2 * node] += self.lazy[node]
            self.tree[2 * node + 1] += self.lazy[node]
            # Propagate lazy value to children's lazy values
            self.lazy[2 * node] += self.lazy[node]
            self.lazy[2 * node + 1] += self.lazy[node]
            # Reset current node's lazy value
            self.lazy[node] = 0

    def update_range(self, l, r, val, node, start, end):
        # Apply `val` to range [l, r]
        # If the current segment [start, end] is completely outside the update range [l, r]
        if r < start or end < l: 
            return
        # If the current segment [start, end] is completely inside the update range [l, r]
        if l <= start and end <= r: 
            self.tree[node] += val
            self.lazy[node] += val
            return

        # Push down lazy updates before recursing
        self._push(node) 

        mid = (start + end) // 2
        # Recursively update children
        self.update_range(l, r, val, 2 * node, start, mid)
        self.update_range(l, r, val, 2 * node + 1, mid + 1, end)
        # After children are updated, update current node's value
        self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])

    def query_max(self, l, r, node, start, end):
        # Query maximum value in range [l, r]
        # If the current segment [start, end] is completely outside the query range [l, r]
        if r < start or end < l: 
            return -float('inf') # Return a very small value for out-of-range queries
        # If the current segment [start, end] is completely inside the query range [l, r]
        if l <= start and end <= r: 
            return self.tree[node]

        # Push down lazy updates before recursing
        self._push(node) 

        mid = (start + end) // 2
        # Recursively query children and return the max
        p1 = self.query_max(l, r, 2 * node, start, mid)
        p2 = self.query_max(l, r, 2 * node + 1, mid + 1, end)
        return max(p1, p2)

# Read input
N = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split()))

# 1. Precompute dp_left: distinct count for A[0...k-1]
# dp_left[k] stores the distinct count of A[0...k-1] (prefix of length k)
freq = {}
distinct_count = 0
dp_left = [0] * (N + 1)
for k in range(N):
    if freq.get(A[k], 0) == 0:
        distinct_count += 1
    freq[A[k]] = freq.get(A[k], 0) + 1
    dp_left[k+1] = distinct_count

# 2. Precompute dp_right: distinct count for A[k...N-1]
# dp_right[k] stores the distinct count of A[k...N-1] (suffix starting at index k)
freq = {}
distinct_count = 0
dp_right = [0] * (N + 1) # dp_right[N] will be 0 for empty suffix
for k in range(N - 1, -1, -1):
    if freq.get(A[k], 0) == 0:
        distinct_count += 1
    freq[A[k]] = freq.get(A[k], 0) + 1
    dp_right[k] = distinct_count

# Segment tree setup
# The segment tree stores values related to the first split point `i`.
# `i` in the problem refers to the split position `A[0...i-1] | A[i...j-1]`.
# Valid `i` values are from 1 to N-2 (ensuring all three parts are non-empty).
# Segment tree indices `0` to `N-3` map to problem `i` values `1` to `N-2`.
# So, segment tree index `st_idx` maps to problem `i` as `st_idx + 1`.
# Initial values for segment tree leaves are `dp_left[i]`.
# For `st_idx = 0`, `i = 1`, so `dp_left[1]`.
# For `st_idx = N-3`, `i = N-2`, so `dp_left[N-2]`.
seg_tree_size = N - 2 # This is the number of valid `i` split points
initial_st_values = [dp_left[i] for i in range(1, N - 1)] # This covers i from 1 to N-2

segment_tree = SegmentTree(seg_tree_size, initial_st_values)
last_occurrence = {} # Maps value -> last seen 0-indexed position in A
max_overall_sum = 0

# Iterate `idx_in_A` as the rightmost index of the middle segment A[i...idx_in_A].
# `idx_in_A` ranges from `1` to `N-2`.
# If `idx_in_A` is `0`, then `j_actual_split_point` (second cut) would be `1`.
# But `j` must be `i+1` at minimum, and `i` must be `1` at minimum, so `j` minimum is `2`.
# So `idx_in_A = j-1` must be at least `1`.
for idx_in_A in range(1, N - 1): # A[1], A[2], ..., A[N-2]
    current_val = A[idx_in_A]
    
    # `prev_idx` is the 0-indexed position where `current_val` last appeared before `idx_in_A`.
    prev_idx = last_occurrence.get(current_val, -1)

    # Calculate the range [l, r] (in terms of segment tree indices) that need a +1 update.
    # An element `A[idx_in_A]` is new in `A[i...idx_in_A]` if its `prev_idx` is less than `i`.
    # So, `i` must be in `[max(1, prev_idx + 1), idx_in_A]`.
    # Convert problem `i` (1-indexed split point) to segment tree `st_idx` (0-indexed).
    # `st_idx = i - 1`.
    st_update_l = max(1, prev_idx + 1) - 1 
    st_update_r = idx_in_A - 1             

    # Perform range update if the range is valid
    if st_update_l <= st_update_r:
        # Check if the update range is within the segment tree's bounds
        # (st_update_l and st_update_r must be within [0, seg_tree_size - 1])
        # st_update_l >= 0 is always true because max(1, ...) - 1 >= 0.
        # st_update_r = idx_in_A - 1, and idx_in_A goes up to N-2, so st_update_r goes up to N-3.
        # N-3 is seg_tree_size - 1. So range is always valid.
        segment_tree.update_range(st_update_l, st_update_r, 1, 1, 0, seg_tree_size - 1)
    
    # Update last_occurrence for current_val's current position
    last_occurrence[current_val] = idx_in_A

    # Now, calculate the total sum for the current `j_actual_split_point`.
    # `j_actual_split_point` is `idx_in_A + 1`.
    # This `j` must be `2 <= j <= N-1`.
    # `idx_in_A` runs from `1` to `N-2`. So `j_actual_split_point` runs from `2` to `N-1`.
    # The first split point `i` must satisfy `1 <= i < j_actual_split_point`, so `1 <= i <= idx_in_A`.
    
    # Convert `i` range `[1, idx_in_A]` to segment tree indices `[0, idx_in_A - 1]`.
    st_query_l = 0
    st_query_r = idx_in_A - 1
    
    # Query for max score in the relevant range of `i` values.
    # A query is valid if st_query_l <= st_query_r and st_query_r is within segment tree size.
    # `st_query_l` is 0. `st_query_r` goes up to `N-3` (when `idx_in_A = N-2`), which is `seg_tree_size - 1`.
    # So `st_query_r < seg_tree_size` is always true if `st_query_l <= st_query_r`.
    if st_query_l <= st_query_r:
        max_score_middle_left = segment_tree.query_max(st_query_l, st_query_r, 1, 0, seg_tree_size - 1)
        
        # Total sum for this `j_actual_split_point` is `max_score_middle_left + dp_right[j_actual_split_point]`
        current_total_sum = max_score_middle_left + dp_right[idx_in_A + 1]
        max_overall_sum = max(max_overall_sum, current_total_sum)

# Print the final maximum sum
print(max_overall_sum)