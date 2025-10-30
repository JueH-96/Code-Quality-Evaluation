import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    
    # Constraints state 1 <= N, Q. So Q is at least 1.
    # x_queries are 1-based element values.
    x_queries = list(map(int, sys.stdin.readline().split()))

    # S_val_history[i] will store |S| after the (i+1)-th query (0-indexed query i)
    S_val_history = [0] * Q 
    
    current_S = set()
    
    # flip_times[k] stores a list of 1-based query indices when element (k+1) was queried.
    # Element values are 1 to N. flip_times is 0-indexed (0 to N-1).
    flip_times = [[] for _ in range(N)]

    for i in range(Q): # Query i (0-indexed), corresponds to problem's (i+1)-th query
        query_val = x_queries[i] # This is x_{i+1} from problem statement
        
        # Store 1-based query index: i+1
        flip_times[query_val - 1].append(i + 1) 
        
        if query_val in current_S:
            current_S.remove(query_val)
        else:
            current_S.add(query_val)
        
        S_val_history[i] = len(current_S) # |S| after (i+1)-th query (1-based index)

    # prefix_sums_S_val[k] stores s_1 + s_2 + ... + s_k
    # where s_m is |S| after m-th query (1-based query m).
    # S_val_history[i] is s_{i+1}.
    # So, prefix_sums_S_val[k] = S_val_history[0] + ... + S_val_history[k-1].
    prefix_sums_S_val = [0] * (Q + 1) # Size Q+1, for P[0] to P[Q]
    for i in range(Q): 
        prefix_sums_S_val[i+1] = prefix_sums_S_val[i] + S_val_history[i]

    # final_A[j] stores result for element j+1
    final_A = [0] * N 

    for j_idx in range(N): # j_idx is 0 to N-1, corresponds to element j_idx+1
        current_element_ans = 0
        is_in_S_currently = False
        # last_entry_query_idx: 1-based query index when element (j_idx+1) was last added to S
        last_entry_query_idx = 0 # Placeholder, will be set to a 1-based query index

        for query_idx_of_flip in flip_times[j_idx]: # query_idx_of_flip is 1-based
            if not is_in_S_currently:
                # Element (j_idx+1) is added to S at query_idx_of_flip.
                is_in_S_currently = True
                last_entry_query_idx = query_idx_of_flip
            else:
                # Element (j_idx+1) is removed from S at query_idx_of_flip.
                # It was in S during interval of queries [last_entry_query_idx, query_idx_of_flip - 1].
                # Add sum of s_k for k in [last_entry_query_idx, query_idx_of_flip - 1].
                # This sum is P[query_idx_of_flip - 1] - P[last_entry_query_idx - 1].
                is_in_S_currently = False
                current_element_ans += prefix_sums_S_val[query_idx_of_flip - 1] - prefix_sums_S_val[last_entry_query_idx - 1]
        
        if is_in_S_currently:
            # Element was last added at last_entry_query_idx and remained in S until query Q (inclusive).
            # It was in S during interval of queries [last_entry_query_idx, Q].
            # Add sum of s_k for k in [last_entry_query_idx, Q].
            # This sum is P[Q] - P[last_entry_query_idx - 1].
            current_element_ans += prefix_sums_S_val[Q] - prefix_sums_S_val[last_entry_query_idx - 1]
            
        final_A[j_idx] = current_element_ans
    
    print(*(final_A))

solve()