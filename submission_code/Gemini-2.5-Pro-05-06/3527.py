import sys

# It's good practice to set higher recursion limit for segment tree if N is large,
# though for N=5e4, log2(N) is about 16, so default limit (1000) is usually fine.
# sys.setrecursionlimit(10**5) 

class FenwickTree:
    def __init__(self, size): # Size is max_val + 1, e.g. if max index is N-1, size is N
        self.tree = [0] * (size + 1) # 1-indexed internal tree

    def update(self, index, delta): # index is 0-based for user
        index += 1 # Convert to 1-indexed
        while index < len(self.tree):
            self.tree[index] += delta
            index += index & (-index)

    def query(self, index): # prefix sum up to index (0-based for user, inclusive)
        index += 1 # Convert to 1-indexed
        s = 0
        while index > 0:
            s += self.tree[index]
            index -= index & (-index)
        return s

    def query_range(self, i, j): # sum from i to j (inclusive, 0-based)
        if i > j: return 0
        return self.query(j) - self.query(i - 1)

class SegmentTree:
    def __init__(self, n, initial_arr=None):
        self.n = n
        self.tree = [0] * (4 * n) 
        if initial_arr:
            self._build(initial_arr, 0, 0, n - 1)

    def _build(self, arr, node_idx, current_L, current_R):
        if current_L == current_R:
            self.tree[node_idx] = arr[current_L]
        else:
            mid = (current_L + current_R) // 2
            self._build(arr, 2 * node_idx + 1, current_L, mid)
            self._build(arr, 2 * node_idx + 2, mid + 1, current_R)
            self.tree[node_idx] = self.tree[2 * node_idx + 1] + self.tree[2 * node_idx + 2]

    def update_val(self, idx_to_update, new_val):
        self._update_val_recursive(0, 0, self.n - 1, idx_to_update, new_val)

    def _update_val_recursive(self, node_idx, current_L, current_R, idx_to_update, new_val):
        if current_L == current_R:
            self.tree[node_idx] = new_val
            return
        mid = (current_L + current_R) // 2
        if idx_to_update <= mid:
            self._update_val_recursive(2 * node_idx + 1, current_L, mid, idx_to_update, new_val)
        else:
            self._update_val_recursive(2 * node_idx + 2, mid + 1, current_R, idx_to_update, new_val)
        self.tree[node_idx] = self.tree[2 * node_idx + 1] + self.tree[2 * node_idx + 2]

    def query_val(self, idx_to_query):
        return self._query_val_recursive(0, 0, self.n - 1, idx_to_query)

    def _query_val_recursive(self, node_idx, current_L, current_R, idx_to_query):
        if current_L == current_R:
            return self.tree[node_idx]
        mid = (current_L + current_R) // 2
        if idx_to_query <= mid:
            return self._query_val_recursive(2 * node_idx + 1, current_L, mid, idx_to_query)
        else:
            return self._query_val_recursive(2 * node_idx + 2, mid + 1, current_R, idx_to_query)

    def _find_zero_recursive(self, node_idx, current_L, current_R, query_L, query_R, find_leftmost):
        if current_R < query_L or current_L > query_R: return None # No overlap
        if self.tree[node_idx] == (current_R - current_L + 1): return None # All ones in node's range
        
        if current_L == current_R: # Leaf node
            return current_L if self.tree[node_idx] == 0 else None

        mid = (current_L + current_R) // 2
        res = None
        if find_leftmost:
            res = self._find_zero_recursive(2 * node_idx + 1, current_L, mid, query_L, query_R, find_leftmost)
            if res is not None: return res
            return self._find_zero_recursive(2 * node_idx + 2, mid + 1, current_R, query_L, query_R, find_leftmost)
        else: # find_rightmost
            res = self._find_zero_recursive(2 * node_idx + 2, mid + 1, current_R, query_L, query_R, find_leftmost)
            if res is not None: return res
            return self._find_zero_recursive(2 * node_idx + 1, current_L, mid, query_L, query_R, find_leftmost)

    # Finds rightmost 0 in A[query_L...query_R]
    def find_rightmost_zero_in_range(self, query_L, query_R):
        if query_L > query_R : return None 
        return self._find_zero_recursive(0, 0, self.n - 1, query_L, query_R, False)

    # Finds leftmost 0 in A[query_L...query_R]
    def find_leftmost_zero_in_range(self, query_L, query_R):
        if query_L > query_R: return None
        return self._find_zero_recursive(0, 0, self.n - 1, query_L, query_R, True)


class Solution:
    def numberOfAlternatingGroups(self, colors: list[int], queries: list[list[int]]) -> list[int]:
        N = len(colors)
        A = [0] * N
        for i in range(N):
            if colors[i] != colors[(i + 1) % N]:
                A[i] = 1
        
        seg_tree_A = SegmentTree(N, A)
        
        # Fenwick trees for lengths of blocks of 1s in A. Max length N.
        # Lengths 1..N map to BIT indices 0..N-1. BIT size is N.
        bit_counts = FenwickTree(N) 
        bit_sum_lengths = FenwickTree(N)

        def get_block_len(s, e, N_val):
            if s == -1 and e == -1: return 0 # Sentinel for no block / empty block
            if s <= e: return e - s + 1
            else: return (N_val - s) + (e + 1)

        def _update_bits(s, e, N_val, delta):
            block_len = get_block_len(s, e, N_val)
            if block_len == 0: return
            
            bit_counts.update(block_len - 1 , delta) # Map length L to index L-1
            bit_sum_lengths.update(block_len - 1, delta * block_len)

        add_block_info = lambda s, e, N_val: _update_bits(s, e, N_val, 1)
        remove_block_info = lambda s, e, N_val: _update_bits(s, e, N_val, -1)

        def get_block_endpoints(idx_A, N_val, st_A):
            # Assumes st_A.query_val(idx_A) == 1
            if st_A.tree[0] == N_val: # All 1s
                return (0, N_val - 1)

            # Find s: first 0 to the left of idx_A (circularly), then +1
            # Search order: (idx_A-1), (idx_A-2), ..., 0, (N-1), (N-2), ..., (idx_A+1)
            zero_before_s = st_A.find_rightmost_zero_in_range(0, (idx_A - 1 + N_val) % N_val)
            if zero_before_s is None: # All 1s in [0 .. idx_A-1]
                 zero_before_s = st_A.find_rightmost_zero_in_range((idx_A + 1) % N_val, N_val - 1)
            s_val = (zero_before_s + 1) % N_val if zero_before_s is not None else idx_A # Should not be None if not all 1s

            # Find e: first 0 to the right of idx_A (circularly), then -1
            # Search order: (idx_A+1), (idx_A+2), ..., N-1, 0, 1, ..., (idx_A-1)
            zero_after_e = st_A.find_leftmost_zero_in_range((idx_A + 1) % N_val, N_val - 1)
            if zero_after_e is None: # All 1s in [idx_A+1 .. N-1]
                zero_after_e = st_A.find_leftmost_zero_in_range(0, (idx_A - 1 + N_val) % N_val)
            e_val = (zero_after_e - 1 + N_val) % N_val if zero_after_e is not None else idx_A
            
            return s_val, e_val

        if seg_tree_A.tree[0] == N and N > 0 : # All 1s
             add_block_info(0, N - 1, N)
        elif seg_tree_A.tree[0] > 0: # Not all 0s, not all 1s
            for i in range(N):
                if seg_tree_A.query_val(i) == 1 and seg_tree_A.query_val((i - 1 + N) % N) == 0:
                    s, e = get_block_endpoints(i, N, seg_tree_A) # s should be i
                    add_block_info(s, e, N)
        
        ans_list = []

        def process_A_update(idx_A, old_A_val, new_A_val):
            if old_A_val == new_A_val: return

            if new_A_val == 1: # 0 -> 1 flip at A[idx_A]
                prev_A_is_1 = seg_tree_A.query_val((idx_A - 1 + N) % N) == 1
                next_A_is_1 = seg_tree_A.query_val((idx_A + 1) % N) == 1
                
                s_b1, e_b1, s_b2, e_b2 = -1,-1,-1,-1 # Sentinels

                if prev_A_is_1: # A[(idx_A-1)%N] is 1. Find its block.
                    # Temporarily set A[idx_A]=0 to correctly find original block B1
                    seg_tree_A.update_val(idx_A, 0) 
                    s_b1, e_b1 = get_block_endpoints((idx_A - 1 + N) % N, N, seg_tree_A)
                    seg_tree_A.update_val(idx_A, 1) # Restore A[idx_A]=1 for further logic
                    remove_block_info(s_b1, e_b1, N)

                if next_A_is_1: # A[(idx_A+1)%N] is 1. Find its block.
                    # Temporarily set A[idx_A]=0 to correctly find original block B2
                    seg_tree_A.update_val(idx_A, 0)
                    s_b2, e_b2 = get_block_endpoints((idx_A + 1) % N, N, seg_tree_A)
                    seg_tree_A.update_val(idx_A, 1)
                    # If B2 is same as B1 (prev_A_is_1 was true, and block wrapped around), don't double remove
                    if not (prev_A_is_1 and s_b1 == s_b2 and e_b1 == e_b2):
                        remove_block_info(s_b2, e_b2, N)
                
                # Add new block(s)
                if prev_A_is_1 and next_A_is_1: # Merge B1 and B2 via idx_A
                    add_block_info(s_b1, e_b2, N) # New merged block (s_B1, e_B2)
                elif prev_A_is_1: # Extend B1
                    add_block_info(s_b1, idx_A, N)
                elif next_A_is_1: # Extend B2
                    add_block_info(idx_A, e_b2, N)
                else: # New block of length 1: (idx_A, idx_A)
                    add_block_info(idx_A, idx_A, N)
            
            else: # 1 -> 0 flip at A[idx_A]
                s_old, e_old = get_block_endpoints(idx_A, N, seg_tree_A) # Get block while A[idx_A] is 1
                remove_block_info(s_old, e_old, N)
                seg_tree_A.update_val(idx_A, 0) # Now A[idx_A] is 0

                # Check if left part forms a new block: (s_old ... idx_A-1)
                len_left_part = get_block_len(s_old, (idx_A - 1 + N) % N, N)
                if not (s_old == idx_A and e_old == idx_A) and len_left_part > 0 : # Original block wasn't just idx_A
                    if seg_tree_A.query_val(s_old) == 1 and seg_tree_A.query_val((idx_A - 1 + N) % N) == 1: # Both ends must be 1
                         add_block_info(s_old, (idx_A - 1 + N) % N, N)
                
                # Check if right part forms a new block: (idx_A+1 ... e_old)
                len_right_part = get_block_len((idx_A + 1) % N, e_old, N)
                if not (s_old == idx_A and e_old == idx_A) and len_right_part > 0 :
                     if seg_tree_A.query_val((idx_A + 1) % N) == 1 and seg_tree_A.query_val(e_old) == 1:
                        add_block_info((idx_A + 1) % N, e_old, N)
        
        for query in queries:
            q_type = query[0]
            if q_type == 1:
                size_L = query[1]
                k = size_L - 1 
                
                # Constraints: 3 <= L <= N-1. So 2 <= k <= N-2.
                # Query BITs for indices k-1 to N-1 (lengths k to N)
                num_blocks_ge_k = bit_counts.query_range(k - 1, N - 1)
                sum_lens_blocks_ge_k = bit_sum_lengths.query_range(k - 1, N - 1)
                
                current_ans = sum_lens_blocks_ge_k - (k - 1) * num_blocks_ge_k
                ans_list.append(max(0, current_ans)) # max(0, ..) defensive, should be >=0

            else: # q_type == 2
                idx_color, new_color_val = query[1], query[2]
                
                if colors[idx_color] == new_color_val: continue
                
                # Store old values of A that will be affected, BEFORE changing colors array
                affected_A_indices = [(idx_color - 1 + N) % N, idx_color]
                old_A_vals = {}
                for idx_A in affected_A_indices:
                    old_A_vals[idx_A] = seg_tree_A.query_val(idx_A)
                
                colors[idx_color] = new_color_val # Actual color update
                
                for idx_A in affected_A_indices:
                    new_A_val_calc = 1 if colors[idx_A] != colors[(idx_A + 1) % N] else 0
                    process_A_update(idx_A, old_A_vals[idx_A], new_A_val_calc)
        return ans_list