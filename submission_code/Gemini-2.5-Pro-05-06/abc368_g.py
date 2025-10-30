import sys

def solve():
    N = int(sys.stdin.readline())
    A_arr_orig = list(map(int, sys.stdin.readline().split()))
    B_arr_orig = list(map(int, sys.stdin.readline().split()))

    A_arr = list(A_arr_orig) # Operates on mutable copies
    B_arr = list(B_arr_orig)

    C_THRESHOLD = 62 
    VALUE_CAP_DP = 4 * 10**18 

    # Segment tree: 1-indexed nodes, stores lists of operations
    # Max N = 10^5. Tree size up to 4*N.
    tree_nodes = [[] for _ in range(4 * N if N > 0 else 1)]


    def safe_add(v, val_add):
        res = v + val_add
        return min(res, VALUE_CAP_DP)

    def safe_mul(v, val_mul):
        if v == 0: return 0
        # B_i >= 1, so val_mul > 0 for choice operations.
        # For B_i=1, v > VALUE_CAP_DP / 1 means v > VALUE_CAP_DP. Overflow if v is already VALUE_CAP_DP.
        # For B_i > 1, standard overflow check.
        if val_mul > 0 and v > VALUE_CAP_DP / val_mul: 
            return VALUE_CAP_DP
        res = v * val_mul
        return min(res, VALUE_CAP_DP)


    def merge_ops(list1, list2):
        if not list1: # If list1 is empty, result is list2 (possibly truncated)
            if len(list2) > C_THRESHOLD:
                return list2[:C_THRESHOLD]
            return list2
        if not list2: # If list2 is empty, result is list1 (already truncated)
            return list1

        res = list1[:] # Crucial copy, as list1 might be a direct reference to a tree node's list
        
        # If last of res (from list1) is sum and first of list2 is sum, combine them
        if res[-1][1] == 0 and list2[0][1] == 0:
            new_sum_val = res[-1][0] + list2[0][0]
            # Sums are also capped. Max sum of A_i's is 10^5 * 10^9 = 10^14, well within VALUE_CAP_DP.
            res[-1] = (min(new_sum_val, VALUE_CAP_DP), 0)
            
            if len(list2) > 1: # If list2 had more elements after the first sum
                 res.extend(list2[1:])
        else: # Otherwise, just append all of list2
            res.extend(list2)
        
        # Truncate if combined list is too long
        if len(res) > C_THRESHOLD:
            return res[:C_THRESHOLD]
        return res

    def build(node_idx, r_low, r_high):
        if r_low == r_high: # Leaf node
            # Cap initial A_i values added to list, though A_i <= 10^9 is far below VALUE_CAP_DP
            current_A = min(A_arr[r_low], VALUE_CAP_DP)
            if B_arr[r_low] == 1:
                tree_nodes[node_idx] = [(current_A, 0)]
            else:
                tree_nodes[node_idx] = [(current_A, B_arr[r_low])]
            return

        mid = (r_low + r_high) // 2
        build(2 * node_idx, r_low, mid)
        build(2 * node_idx + 1, mid + 1, r_high)
        tree_nodes[node_idx] = merge_ops(tree_nodes[2 * node_idx], tree_nodes[2 * node_idx + 1])

    def update(node_idx, r_low, r_high, target_idx, val_new, type_update):
        if r_low == r_high: # Leaf node
            if type_update == 1: 
                A_arr[target_idx] = val_new
            else: 
                B_arr[target_idx] = val_new
            
            current_A = min(A_arr[target_idx], VALUE_CAP_DP)
            if B_arr[target_idx] == 1:
                tree_nodes[node_idx] = [(current_A, 0)]
            else:
                tree_nodes[node_idx] = [(current_A, B_arr[target_idx])]
            return

        mid = (r_low + r_high) // 2
        if target_idx <= mid:
            update(2 * node_idx, r_low, mid, target_idx, val_new, type_update)
        else:
            update(2 * node_idx + 1, mid + 1, r_high, target_idx, val_new, type_update)
        
        tree_nodes[node_idx] = merge_ops(tree_nodes[2 * node_idx], tree_nodes[2 * node_idx + 1])

    # Query for the list of operations for range [query_l, query_r]
    def query_ops_recursive(node_idx, r_low, r_high, query_l, query_r):
        # If current segment [r_low, r_high] is outside query range [query_l, query_r]
        if query_l > r_high or query_r < r_low: 
            return []
        # If current segment is completely inside query range
        if query_l <= r_low and r_high <= query_r: 
            return tree_nodes[node_idx]
        
        mid = (r_low + r_high) // 2
        res_left = query_ops_recursive(2 * node_idx, r_low, mid, query_l, query_r)
        res_right = query_ops_recursive(2 * node_idx + 1, mid + 1, r_high, query_l, query_r)
        
        # merge_ops handles cases where one or both lists are empty
        return merge_ops(res_left, res_right)

    # Initial build of the segment tree
    if N > 0 : build(1, 0, N - 1)

    Q_count = int(sys.stdin.readline())
    output_buffer = [] 

    for _ in range(Q_count):
        query_parts = list(map(int, sys.stdin.readline().split()))
        type_q = query_parts[0]

        if type_q == 1: # Update A_i
            idx, x = query_parts[1], query_parts[2]
            if N > 0: update(1, 0, N - 1, idx - 1, x, 1) # Use 0-indexed idx
        elif type_q == 2: # Update B_i
            idx, x = query_parts[1], query_parts[2]
            if N > 0: update(1, 0, N - 1, idx - 1, x, 2) # Use 0-indexed idx
        else: # Type 3 query
            l, r = query_parts[1], query_parts[2]
            
            if N == 0: # Should not happen based on constraints N >= 1
                output_buffer.append("0")
                continue

            final_ops_list = query_ops_recursive(1, 0, N - 1, l - 1, r - 1) # Use 0-indexed range
            
            current_v = 0
            for op_val, op_b_val in final_ops_list:
                if op_b_val == 0: # This is a sum operation (S, 0)
                    current_v = safe_add(current_v, op_val)
                else: # This is a choice operation (A_k, B_k)
                    v_if_add = safe_add(current_v, op_val)
                    v_if_mul = safe_mul(current_v, op_b_val)
                    current_v = max(v_if_add, v_if_mul)
            
            # Final answer guaranteed to be <= 10^18
            output_buffer.append(str(min(current_v, 10**18)))

    sys.stdout.write('
'.join(output_buffer))
    if output_buffer: # Add trailing newline only if there's output
        sys.stdout.write('
')

solve()