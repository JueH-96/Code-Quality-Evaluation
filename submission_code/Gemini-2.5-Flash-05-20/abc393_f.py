import sys
from bisect import bisect_left, bisect_right

class SegmentTree:
    def __init__(self, size):
        self.size = size
        # Using 1-based indexing for tree nodes, so 4 * size is a safe upper bound for array size
        # tree[0] is unused, root is at tree[1]
        self.tree = [0] * (4 * size) 

    def update(self, node_idx, start, end, target_idx, val):
        """
        Updates the segment tree at target_idx with value `val`.
        The value at target_idx will be max(current_value, val).
        """
        if start == end: # Leaf node reached
            self.tree[node_idx] = max(self.tree[node_idx], val)
            return

        mid = (start + end) // 2
        if start <= target_idx <= mid:
            # target_idx is in the left child's range
            self.update(2 * node_idx, start, mid, target_idx, val)
        else:
            # target_idx is in the right child's range
            self.update(2 * node_idx + 1, mid + 1, end, target_idx, val)
        
        # Update current node's value based on its children
        self.tree[node_idx] = max(self.tree[2 * node_idx], self.tree[2 * node_idx + 1])

    def query(self, node_idx, start, end, query_l, query_r):
        """
        Queries the maximum value in the range [query_l, query_r].
        """
        # If the current segment node's range is outside the query range
        if query_r < start or end < query_l:
            return 0 # No overlap, return identity for max (0)
        
        # If the current segment node's range is completely within the query range
        if query_l <= start and end <= query_r:
            return self.tree[node_idx]
        
        # Partial overlap, recurse on children
        mid = (start + end) // 2
        p1 = self.query(2 * node_idx, start, mid, query_l, query_r)
        p2 = self.query(2 * node_idx + 1, mid + 1, end, query_l, query_r)
        return max(p1, p2)

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    queries_input = []
    # Collect all unique values for coordinate compression
    all_values_set = set(A)
    for i in range(Q):
        R, X = map(int, sys.stdin.readline().split())
        queries_input.append((R, X, i)) # Store (R_i, X_i, original_index_i)
        all_values_set.add(X) # Add X_i to values for compression

    # Sort unique values to create mapping
    all_values = sorted(list(all_values_set))
    value_to_coord = {val: i for i, val in enumerate(all_values)}
    M = len(all_values) # Size of compressed coordinate space

    st = SegmentTree(M)
    answers = [0] * Q # To store results in original query order

    # Sort queries by R_i to process them offline
    # Python's sort is stable; if R_i are equal, original relative order is preserved
    # but for (R, X, original_idx) tuples, it sorts by R, then X, then original_idx
    queries_input.sort() 

    current_query_idx_in_sorted_list = 0
    
    # Iterate through A elements (0-indexed j from 0 to N-1)
    for j in range(N): 
        val_A = A[j]
        
        # 1. Calculate LIS length ending with A[j] and update segment tree
        coord_A = value_to_coord[val_A]
        
        # Query for the maximum LIS length that can be extended by A[j]
        # This is the max LIS length ending with a value strictly less than A[j].
        # In compressed coordinates, this corresponds to querying range [0, coord_A - 1].
        max_prev_lis_len = st.query(1, 0, M - 1, 0, coord_A - 1)
        
        # The LIS length ending with A[j] is 1 + (max_prev_lis_len, or 0 if no such element)
        current_lis_len = max_prev_lis_len + 1
        
        # Update the segment tree at coord_A with this new LIS length
        # We take max because multiple paths might lead to the same value,
        # but we only care about the longest one.
        st.update(1, 0, M - 1, coord_A, current_lis_len)

        # 2. Answer all pending queries whose R_i matches current (j + 1)
        # Note: R_i is 1-indexed, j is 0-indexed.
        while current_query_idx_in_sorted_list < Q and \
              queries_input[current_query_idx_in_sorted_list][0] == j + 1:
            
            R_i, X_i, original_idx = queries_input[current_query_idx_in_sorted_list]
            
            # Find the largest compressed coordinate `c_X_i` such that `all_values[c_X_i] <= X_i`.
            # `bisect_right(list, val)` returns an insertion point `k` such that all `list[i]` for `i < k` are `<= val`.
            # So, `list[k-1]` is the largest element less than or equal to `val`.
            coord_X_i = bisect_right(all_values, X_i) - 1
            
            # Query the segment tree for the maximum LIS length among values <= X_i.
            # This corresponds to querying range [0, coord_X_i].
            # If coord_X_i is -1 (e.g., X_i is smaller than all values in all_values), 
            # the query range becomes [0, -1], and the segment tree query function will correctly return 0.
            ans = st.query(1, 0, M - 1, 0, coord_X_i)
            answers[original_idx] = ans
            
            current_query_idx_in_sorted_list += 1
    
    # Print all answers
    for ans in answers:
        sys.stdout.write(str(ans) + '
')

# Call the solver function
solve()