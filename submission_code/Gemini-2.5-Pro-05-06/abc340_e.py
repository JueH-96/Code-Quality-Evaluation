import sys

# Set higher recursion limit for safety, though default might be okay for N=2e5
sys.setrecursionlimit(10**6) 

# Standard input reading functions
def get_ints(): return map(int, sys.stdin.readline().split())
def get_list(): return list(map(int, sys.stdin.readline().split()))

class SegTree:
    def __init__(self, n, initial_A):
        self.n = n
        self.tree = [0] * (4 * n)  # Stores sums for ranges
        self.lazy = [0] * (4 * n)  # Stores lazy values for range additions
        self._build(initial_A, 1, 0, n - 1)

    def _build(self, arr, node_idx, current_L, current_R):
        if current_L == current_R:
            self.tree[node_idx] = arr[current_L]
        else:
            mid = (current_L + current_R) // 2
            left_child_idx = 2 * node_idx
            right_child_idx = 2 * node_idx + 1
            self._build(arr, left_child_idx, current_L, mid)
            self._build(arr, right_child_idx, mid + 1, current_R)
            self.tree[node_idx] = self.tree[left_child_idx] + self.tree[right_child_idx]

    def _push(self, node_idx, current_L, current_R):
        if self.lazy[node_idx] != 0 and current_L != current_R: # Only push if not leaf and lazy is non-zero
            mid = (current_L + current_R) // 2
            left_child_idx = 2 * node_idx
            right_child_idx = 2 * node_idx + 1
            
            # Apply lazy to children's sum
            self.tree[left_child_idx] += self.lazy[node_idx] * (mid - current_L + 1)
            self.tree[right_child_idx] += self.lazy[node_idx] * (current_R - mid)
            
            # Pass lazy to children's lazy tag
            self.lazy[left_child_idx] += self.lazy[node_idx]
            self.lazy[right_child_idx] += self.lazy[node_idx]
            
            self.lazy[node_idx] = 0

    def _update_range_recursive(self, node_idx, current_L, current_R, query_L, query_R, val_to_add):
        self._push(node_idx, current_L, current_R) 
        
        if current_L > current_R or current_L > query_R or current_R < query_L: 
            return
        
        if query_L <= current_L and current_R <= query_R: 
            self.tree[node_idx] += val_to_add * (current_R - current_L + 1)
            if current_L != current_R: 
                self.lazy[2 * node_idx] += val_to_add
                self.lazy[2 * node_idx + 1] += val_to_add
            return
        
        mid = (current_L + current_R) // 2
        left_child_idx = 2 * node_idx
        right_child_idx = 2 * node_idx + 1
        self._update_range_recursive(left_child_idx, current_L, mid, query_L, query_R, val_to_add)
        self._update_range_recursive(right_child_idx, mid + 1, current_R, query_L, query_R, val_to_add)
        
        self.tree[node_idx] = self.tree[left_child_idx] + self.tree[right_child_idx]

    def _query_range_recursive(self, node_idx, current_L, current_R, query_L, query_R):
        if current_L > current_R or current_L > query_R or current_R < query_L: 
            return 0
        
        self._push(node_idx, current_L, current_R)
        
        if query_L <= current_L and current_R <= query_R: 
            return self.tree[node_idx]
        
        mid = (current_L + current_R) // 2
        p1 = self._query_range_recursive(2 * node_idx, current_L, mid, query_L, query_R)
        p2 = self._query_range_recursive(2 * node_idx + 1, mid + 1, current_R, query_L, query_R)
        return p1 + p2

    def update_range(self, l, r, val_to_add):
        if l > r : return # Handle empty range; problem constraints likely prevent this for r_mod distribution
        self._update_range_recursive(1, 0, self.n - 1, l, r, val_to_add)

    def query_point(self, idx):
        return self._query_range_recursive(1, 0, self.n - 1, idx, idx)


def main():
    N, M = get_ints()
    A = get_list()
    B_ops = get_list() # These are 0-indexed according to problem (box i, for i=0 to N-1)

    seg_tree = SegTree(N, A)
    offset_all = 0

    for k_op_idx in range(M):
        b_box_idx = B_ops[k_op_idx]

        s_b_val_in_tree = seg_tree.query_point(b_box_idx)
        h_balls = s_b_val_in_tree + offset_all

        if h_balls == 0:
            continue

        q_div = h_balls // N
        r_mod = h_balls % N
        
        val_to_add_for_emptying = -offset_all - s_b_val_in_tree
        seg_tree.update_range(b_box_idx, b_box_idx, val_to_add_for_emptying)
        
        offset_all += q_div

        if r_mod > 0:
            idx_start_distribute = (b_box_idx + 1) % N
            idx_end_distribute = (b_box_idx + r_mod) % N
            
            if idx_start_distribute <= idx_end_distribute:
                seg_tree.update_range(idx_start_distribute, idx_end_distribute, 1)
            else: # Wraps around
                seg_tree.update_range(idx_start_distribute, N - 1, 1)
                seg_tree.update_range(0, idx_end_distribute, 1)
                
    results = []
    for i in range(N):
        results.append(seg_tree.query_point(i) + offset_all)
    
    sys.stdout.write(" ".join(map(str, results)) + "
")

if __name__ == '__main__':
    main()