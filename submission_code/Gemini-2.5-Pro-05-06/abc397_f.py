import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # L_arr[k]: distinct count in A[0...k]
    L_arr = [0] * N
    seen_L = set()
    current_distinct_L = 0
    for k in range(N):
        if A[k] not in seen_L:
            current_distinct_L += 1
            seen_L.add(A[k])
        L_arr[k] = current_distinct_L
    
    # L_val_for_s_init[s_py]: distinct count for A[0...s_py-1] (first segment)
    L_val_for_s_init = [0] * N 
    for s_py in range(1, N):
        L_val_for_s_init[s_py] = L_arr[s_py-1]

    # R_arr[k]: distinct count in A[k...N-1]
    R_arr = [0] * N
    seen_R = set()
    current_distinct_R = 0
    for k in range(N - 1, -1, -1):
        if A[k] not in seen_R:
            current_distinct_R += 1
            seen_R.add(A[k])
        R_arr[k] = current_distinct_R
    
    # R_val_for_e_py_suffix[e_py]: distinct count for A[e_py+1...N-1] (third segment)
    R_val_for_e_py_suffix = [0] * N
    for e_py in range(N - 1): 
        R_val_for_e_py_suffix[e_py] = R_arr[e_py+1]
    
    # Segment Tree
    st_n_leaves = 1
    while st_n_leaves < N:
        st_n_leaves *= 2
    
    st_tree = [-float('inf')] * (2 * st_n_leaves) # Use -inf for initial max values
    st_lazy = [0] * (2 * st_n_leaves)

    for i in range(N):
        st_tree[st_n_leaves + i] = L_val_for_s_init[i]
    # Indices from N to st_n_leaves-1 remain -inf, correctly out of query consideration.

    for i in range(st_n_leaves - 1, 0, -1):
        st_tree[i] = max(st_tree[i * 2], st_tree[i * 2 + 1])

    def _st_push(node_idx):
        if st_lazy[node_idx] != 0 and node_idx < st_n_leaves: 
            st_tree[node_idx * 2] += st_lazy[node_idx]
            st_tree[node_idx * 2 + 1] += st_lazy[node_idx]
            st_lazy[node_idx * 2] += st_lazy[node_idx]
            st_lazy[node_idx * 2 + 1] += st_lazy[node_idx]
            st_lazy[node_idx] = 0

    def _st_update_range(node_idx, current_L, current_R, query_L, query_R, val_to_add):
        if query_L <= current_L and current_R <= query_R: 
            st_tree[node_idx] += val_to_add
            st_lazy[node_idx] += val_to_add
            return
        
        if current_R < query_L or query_R < current_L or query_L > query_R : # Added query_L > query_R check
            return

        _st_push(node_idx) 
        
        mid = (current_L + current_R) // 2
        _st_update_range(node_idx * 2, current_L, mid, query_L, query_R, val_to_add)
        _st_update_range(node_idx * 2 + 1, mid + 1, current_R, query_L, query_R, val_to_add)
        
        st_tree[node_idx] = max(st_tree[node_idx * 2], st_tree[node_idx * 2 + 1])

    def _st_query_max(node_idx, current_L, current_R, query_L, query_R):
        if query_L <= current_L and current_R <= query_R:
            return st_tree[node_idx]
        
        if current_R < query_L or query_R < current_L or query_L > query_R : # Added query_L > query_R check
            return -float('inf') 

        _st_push(node_idx)
        
        mid = (current_L + current_R) // 2
        res_L = _st_query_max(node_idx * 2, current_L, mid, query_L, query_R)
        res_R = _st_query_max(node_idx * 2 + 1, mid + 1, current_R, query_L, query_R)
        return max(res_L, res_R)

    pos_val_to_idx = [-1] * (N + 1) 
    max_total_distinct_count = 0 

    for e_py in range(N):
        val_at_e_py = A[e_py]
        prev_occurrence_idx = pos_val_to_idx[val_at_e_py]

        update_L_s_py, update_R_s_py = prev_occurrence_idx + 1, e_py
        # Only update if range is valid: update_L_s_py <= update_R_s_py
        # This check is implicitly handled by _st_update_range if query_L > query_R it returns.
        _st_update_range(1, 0, st_n_leaves - 1, update_L_s_py, update_R_s_py, 1)
        
        pos_val_to_idx[val_at_e_py] = e_py

        if e_py >= 1 and e_py <= N - 2:
            query_L_s_py, query_R_s_py = 1, e_py
            
            current_max_L_plus_M = _st_query_max(1, 0, st_n_leaves - 1, query_L_s_py, query_R_s_py)
            
            third_part_distinct_count = R_val_for_e_py_suffix[e_py]
            
            if current_max_L_plus_M > -float('inf'):
                 max_total_distinct_count = max(max_total_distinct_count, current_max_L_plus_M + third_part_distinct_count)
            
    sys.stdout.write(str(max_total_distinct_count) + "
")

main()