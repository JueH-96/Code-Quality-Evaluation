import sys

# Segment tree functions (using _-prefixed globals for state)
_st_sum = []
_st_lazy = []
_tree_size = 0 # Power of 2, >= N_orig

def _build_seg_tree(arr_initial, node_idx, current_range_start, current_range_end, N_orig):
    global _st_sum, _st_lazy 
    if current_range_start == current_range_end:
        if current_range_start < N_orig:
            _st_sum[node_idx] = arr_initial[current_range_start]
        else:
            _st_sum[node_idx] = 0 # Segments outside actual string are treated as 0s
        return
    
    mid = (current_range_start + current_range_end) // 2
    _build_seg_tree(arr_initial, 2 * node_idx, current_range_start, mid, N_orig)
    _build_seg_tree(arr_initial, 2 * node_idx + 1, mid + 1, current_range_end, N_orig)
    _st_sum[node_idx] = _st_sum[2 * node_idx] + _st_sum[2 * node_idx + 1]

def _push_lazy_tag(node_idx, current_range_start, current_range_end):
    global _st_sum, _st_lazy
    if _st_lazy[node_idx] != -1: # -1 indicates no pending lazy update
        val_to_set = _st_lazy[node_idx]
        elements_in_range = (current_range_end - current_range_start + 1)
        _st_sum[node_idx] = val_to_set * elements_in_range
        
        if current_range_start != current_range_end: # Not a leaf node, propagate to children
            _st_lazy[2 * node_idx] = val_to_set
            _st_lazy[2 * node_idx + 1] = val_to_set
        _st_lazy[node_idx] = -1 # Clear lazy tag for current node

def _update_range_seg_tree(node_idx, current_range_start, current_range_end, update_L, update_R, value_to_set):
    global _st_sum, _st_lazy
    _push_lazy_tag(node_idx, current_range_start, current_range_end)

    if current_range_start > current_range_end or current_range_start > update_R or current_range_end < update_L: # No overlap
        return

    if update_L <= current_range_start and current_range_end <= update_R: # Current segment fully within update range
        _st_lazy[node_idx] = value_to_set
        _push_lazy_tag(node_idx, current_range_start, current_range_end) # Apply update
        return

    mid = (current_range_start + current_range_end) // 2
    _update_range_seg_tree(2 * node_idx, current_range_start, mid, update_L, update_R, value_to_set)
    _update_range_seg_tree(2 * node_idx + 1, mid + 1, current_range_end, update_L, update_R, value_to_set)
    _st_sum[node_idx] = _st_sum[2 * node_idx] + _st_sum[2 * node_idx + 1]

def _query_sum_seg_tree(node_idx, current_range_start, current_range_end, query_L, query_R):
    global _st_sum, _st_lazy
    if current_range_start > current_range_end or current_range_start > query_R or current_range_end < query_L: # No overlap
        return 0
    
    _push_lazy_tag(node_idx, current_range_start, current_range_end)

    if query_L <= current_range_start and current_range_end <= query_R: # Current segment fully within query range
        return _st_sum[node_idx]
    
    mid = (current_range_start + current_range_end) // 2
    sum_left_child = _query_sum_seg_tree(2 * node_idx, current_range_start, mid, query_L, query_R)
    sum_right_child = _query_sum_seg_tree(2 * node_idx + 1, mid + 1, current_range_end, query_L, query_R)
    return sum_left_child + sum_right_child


def solve():
    global _st_sum, _st_lazy, _tree_size

    N, X, Y = map(int, sys.stdin.readline().split())
    S_str = sys.stdin.readline().strip()
    T_str = sys.stdin.readline().strip()

    S_list = [int(c) for c in S_str]
    T_list = [int(c) for c in T_str]
    
    _tree_size = 1
    while _tree_size < N:
        _tree_size *= 2
    
    _st_sum = [0] * (2 * _tree_size)
    _st_lazy = [-1] * (2 * _tree_size) # -1: no lazy op, 0: set to 0, 1: set to 1

    _build_seg_tree(S_list, 1, 0, _tree_size - 1, N)

    for i in range(N): # Iterate 0-indexed
        # Get current S_i from segment tree
        current_s_i = _query_sum_seg_tree(1, 0, _tree_size - 1, i, i)
        target_t_i = T_list[i]

        if current_s_i == target_t_i:
            continue
        
        # If S_i != T_i, an operation starting at i is needed.
        # Check if operation fits: i to i + (X+Y) - 1 must be < N
        if i + X + Y - 1 >= N:
            print("No") # Operation would exceed string bounds
            return
        
        if current_s_i == 0 and target_t_i == 1: # Need S_i to become 1: Use Op A (0^X 1^Y -> 1^Y 0^X)
            # Check S[i...i+X-1] == 0^X
            sum_block1 = _query_sum_seg_tree(1, 0, _tree_size - 1, i, i + X - 1)
            if sum_block1 != 0: # Not all zeros
                print("No")
                return
            
            # Check S[i+X...i+X+Y-1] == 1^Y
            sum_block2 = _query_sum_seg_tree(1, 0, _tree_size - 1, i + X, i + X + Y - 1)
            if sum_block2 != Y: # Not all ones (or wrong count of ones)
                print("No")
                return

            # Perform Op A
            _update_range_seg_tree(1, 0, _tree_size - 1, i, i + Y - 1, 1)       # First Y chars become 1
            _update_range_seg_tree(1, 0, _tree_size - 1, i + Y, i + Y + X - 1, 0) # Next X chars become 0

        elif current_s_i == 1 and target_t_i == 0: # Need S_i to become 0: Use Op B (1^Y 0^X -> 0^X 1^Y)
            # Check S[i...i+Y-1] == 1^Y
            sum_block1 = _query_sum_seg_tree(1, 0, _tree_size - 1, i, i + Y - 1)
            if sum_block1 != Y: # Not all ones
                print("No")
                return

            # Check S[i+Y...i+Y+X-1] == 0^X
            sum_block2 = _query_sum_seg_tree(1, 0, _tree_size - 1, i + Y, i + Y + X - 1)
            if sum_block2 != 0: # Not all zeros
                print("No")
                return
            
            # Perform Op B
            _update_range_seg_tree(1, 0, _tree_size - 1, i, i + X - 1, 0)       # First X chars become 0
            _update_range_seg_tree(1, 0, _tree_size - 1, i + X, i + X + Y - 1, 1) # Next Y chars become 1
        
    print("Yes")

sys.setrecursionlimit(2 * 10**5) # N can be up to 5e5, tree depth logN. Max segtree size ~2N.
                                  # Depth is log(_tree_size), so log(2N). Max depth around 20 for N=5e5.
                                  # Limit 2*10^5 seems generous, but 10^6 is safer for deeper constructs or overhead.
                                  # Typical competitive programming setup uses 10^6 or more.
solve()