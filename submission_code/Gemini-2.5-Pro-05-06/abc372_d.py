import sys

# It's good practice to increase recursion limit for segment tree operations on large N
sys.setrecursionlimit(2 * 10**5 + 500) # Max N is 2*10^5, tree depth O(log N), but query path might be deeper if not careful

def solve():
    N = int(sys.stdin.readline())
    H_list = list(map(int, sys.stdin.readline().split()))

    if N == 0:
        sys.stdout.write("
")
        return
    if N == 1:
        sys.stdout.write("0
")
        return

    tree_data = [[] for _ in range(4 * N)]
    tree_max_val = [0] * (4 * N)

    # Merges lists from left and right children.
    # list1_records: records from left child (if its segment starts with max_prev=0)
    # max1_val: max H value in left child's entire segment
    # list2_records: records from right child (if its segment starts with max_prev=0)
    def merge_data_lists(list1_records_orig, max1_val, list2_records_orig):
        # `listX_records_orig` is in format [(h,c), ...], h ascending, c descending.
        # These are records if current_max=0 FOR THAT SEGMENT.
        
        # Step 1: Extract pure record values from children, considering max1_val for right child.
        # These are the record values if this merged segment starts with max_prev=0.
        actual_record_values = []
        
        current_max_in_merged_segment = 0
        for h_val, _ in list1_records_orig: # Iterate through records of left child
            if h_val > current_max_in_merged_segment:
                actual_record_values.append(h_val)
                current_max_in_merged_segment = h_val
        
        # For right child records, they must also be greater than overall max from left child (max1_val)
        # And also greater than any subsequent records from left child that were chosen.
        # More simply, they must be greater than current_max_in_merged_segment (which is now max1_val after left pass).
        
        # The `current_max_in_merged_segment` after processing list1_records_orig
        # is indeed `max1_val` because all records from left are <= `max1_val`, and the last one is `max1_val` if `max1_val` itself is a record.
        # Wait, no. `current_max_in_merged_segment` would be the largest record value from list1. This is `max1_val` if `list1_records_orig` is not empty.
        # If `list1_records_orig` is empty, it should be 0.
        # The variable `max1_val` is the true max of H values in left segment.

        effective_threshold_for_right = max1_val # This is correct.
        
        for h_val, _ in list2_records_orig: # Iterate through records of right child
            if h_val > effective_threshold_for_right: # Must be greater than ANY H in left part
                if h_val > current_max_in_merged_segment: # And also a new record for the combined list
                     actual_record_values.append(h_val)
                     current_max_in_merged_segment = h_val
        
        # Step 2: Convert `actual_record_values` (which is sorted ascending) to the (h, count) format.
        merged_list_final = []
        num_total_records = len(actual_record_values)
        for i in range(num_total_records):
            merged_list_final.append((actual_record_values[i], num_total_records - i))
            
        return merged_list_final

    def build(node_idx, L, R):
        if L == R:
            tree_data[node_idx] = [(H_list[L], 1)]
            tree_max_val[node_idx] = H_list[L]
        else:
            mid = (L + R) // 2
            build(2 * node_idx, L, mid)
            build(2 * node_idx + 1, mid + 1, R)
            
            tree_max_val[node_idx] = max(tree_max_val[2*node_idx], tree_max_val[2*node_idx+1])
            tree_data[node_idx] = merge_data_lists(
                tree_data[2*node_idx], tree_max_val[2*node_idx],
                tree_data[2*node_idx+1]
            )

    def query(node_idx, current_L, current_R, query_L, query_R, X_threshold):
        if current_L > query_R or current_R < query_L or query_L > query_R:
            return 0, 0 

        if query_L <= current_L and current_R <= query_R:
            data_list = tree_data[node_idx]
            
            # Binary search for first (h, c) where h > X_threshold
            low, high = 0, len(data_list) - 1
            found_list_idx = len(data_list) # If no element > X_threshold, corresponds to inserting at end
            
            while low <= high:
                mid_list_idx = (low + high) // 2
                if data_list[mid_list_idx][0] > X_threshold:
                    found_list_idx = mid_list_idx
                    high = mid_list_idx - 1
                else:
                    low = mid_list_idx + 1
            
            num_added_records = 0
            if found_list_idx < len(data_list):
                num_added_records = data_list[found_list_idx][1]
            
            return num_added_records, tree_max_val[node_idx]

        mid = (current_L + current_R) // 2
        
        count_L, max_val_L_queried_part = query(2*node_idx, current_L, mid, query_L, query_R, X_threshold)
        
        threshold_for_R = X_threshold
        # max_val_L_queried_part is max H in (queried part of left child). 
        # If queried part of left child is empty, max_val_L_queried_part will be 0.
        threshold_for_R = max(X_threshold, max_val_L_queried_part) 
        
        count_R, max_val_R_queried_part = query(2*node_idx + 1, mid + 1, current_R, query_L, query_R, threshold_for_R)
        
        # Overall max in the queried parts that fall into [current_L, current_R] for this node
        # This is max of H values from those specific indices.
        overall_max_in_queried_intersection = max(max_val_L_queried_part, max_val_R_queried_part)
            
        return count_L + count_R, overall_max_in_queried_intersection

    build(1, 0, N - 1)
    
    ans = [0] * N
    for i in range(N):
        if i == N - 1:
            ans[i] = 0
        else:
            # Query for records in H_list[i+1 ... N-1] with initial X_threshold = 0
            count, _ = query(1, 0, N-1, i + 1, N-1, 0)
            ans[i] = count
            
    sys.stdout.write(" ".join(map(str, ans)) + "
")

solve()