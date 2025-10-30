import sys

class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        # self.tree stores maximums in ranges. Values are 0 or 1.
        # 1 means a bad pair (S_i = S_{i+1}) exists in the range.
        # 0 means all pairs are good (S_i != S_{i+1}) in the range.
        
        # The main solve() function ensures N > 1, so self.n = len(B_arr) = N-1 >= 1.
        self.tree = [0] * (4 * self.n) 
        self._build(arr, 0, 0, self.n - 1)

    def _build(self, arr, node_idx, start, end):
        if start == end:
            self.tree[node_idx] = arr[start]
        else:
            mid = (start + end) // 2
            left_child_idx = 2 * node_idx + 1
            right_child_idx = 2 * node_idx + 2
            self._build(arr, left_child_idx, start, mid)
            self._build(arr, right_child_idx, mid + 1, end)
            self.tree[node_idx] = max(self.tree[left_child_idx], self.tree[right_child_idx])

    def _toggle_update_recursive(self, node_idx, start, end, idx_to_toggle):
        if start == end: # Leaf node
            self.tree[node_idx] = 1 - self.tree[node_idx] # Flip 0 to 1 or 1 to 0
            return

        mid = (start + end) // 2
        left_child_idx = 2 * node_idx + 1
        right_child_idx = 2 * node_idx + 2
        
        if idx_to_toggle <= mid:
            self._toggle_update_recursive(left_child_idx, start, mid, idx_to_toggle)
        else:
            self._toggle_update_recursive(right_child_idx, mid + 1, end, idx_to_toggle)
        
        self.tree[node_idx] = max(self.tree[left_child_idx], self.tree[right_child_idx])

    def toggle_update(self, idx_to_toggle):
        # Assumes idx_to_toggle is a valid 0-based index for B_arr.
        self._toggle_update_recursive(0, 0, self.n - 1, idx_to_toggle)

    def _query_recursive(self, node_idx, current_start, current_end, query_L, query_R):
        if query_L > query_R: # Empty query range
            return 0 
            
        if current_end < query_L or current_start > query_R: # Current segment outside query range
            return 0
        
        if query_L <= current_start and current_end <= query_R: # Current segment inside query range
            return self.tree[node_idx]
        
        mid = (current_start + current_end) // 2
        left_child_idx = 2 * node_idx + 1
        right_child_idx = 2 * node_idx + 2
        
        max_in_left = self._query_recursive(left_child_idx, current_start, mid, query_L, query_R)
        max_in_right = self._query_recursive(right_child_idx, mid + 1, current_end, query_L, query_R)
        
        return max(max_in_left, max_in_right)

    def query(self, query_L, query_R):
        # Assumes query_L, query_R are 0-based indices for B_arr or define an empty range.
        return self._query_recursive(0, 0, self.n - 1, query_L, query_R)

def solve():
    input_reader = sys.stdin.readline 
    
    N, Q = map(int, input_reader().split())
    S_str = input_reader().strip()

    if N == 1:
        for _ in range(Q):
            query_line = input_reader().split()
            query_type = int(query_line[0])
            if query_type == 2:
                sys.stdout.write("Yes
")
        return

    S_int = [int(c) for c in S_str] 
    
    B_arr_len = N - 1
    B_arr = [0] * B_arr_len
    for i in range(B_arr_len):
        if S_int[i] == S_int[i+1]:
            B_arr[i] = 1
    
    seg_tree = SegmentTree(B_arr)
    
    for _ in range(Q):
        query_line = input_reader().split()
        query_type = int(query_line[0])
        L_s = int(query_line[1]) # 1-based index for S string
        R_s = int(query_line[2]) # 1-based index for S string

        if query_type == 1: # Flip operation
            if L_s > 1:
                # B_arr index for pair (S_{L-1}, S_L) is (L-1)-1 = L-2 (0-based).
                idx_b_to_toggle = L_s - 2 
                seg_tree.toggle_update(idx_b_to_toggle)
            
            if R_s < N:
                # B_arr index for pair (S_R, S_{R+1}) is R-1 (0-based).
                idx_b_to_toggle = R_s - 1
                seg_tree.toggle_update(idx_b_to_toggle)
        else: # query_type == 2, Check goodness
            # Substring S[L_s..R_s]. We check B_arr entries for pairs (S_i, S_{i+1})
            # where L_s <= i <= R_s-1.
            # These correspond to B_arr indices from (L_s)-1 to (R_s-1)-1.
            # Query B_arr range [L_s-1, R_s-2].
            
            query_L_b = L_s - 1 
            query_R_b = R_s - 2
            
            max_val_in_B_range = seg_tree.query(query_L_b, query_R_b)
            
            if max_val_in_B_range == 0:
                sys.stdout.write("Yes
")
            else:
                sys.stdout.write("No
")

solve()