import sys

def solve():
    # Read N and Q
    N, Q = map(int, sys.stdin.readline().split())
    # Read the query sequence x_i
    x_queries = list(map(int, sys.stdin.readline().split()))

    # --- Step 1: Process queries to build toggle_times and S_sizes ---
    
    # toggle_times[j-1] will store a list of 1-indexed query indices
    # where element j was toggled (added to or removed from S).
    toggle_times = [[] for _ in range(N)]

    # S_sizes[i] will store the size of set S after the i-th query.
    # We use 1-based indexing for queries, so S_sizes has size Q+1.
    S_sizes = [0] * (Q + 1)

    # current_S tracks the elements currently in the set S.
    current_S = set()

    # Iterate through each query x_i
    for i in range(Q):
        x = x_queries[i]
        
        # Toggle x in current_S
        if x in current_S:
            current_S.remove(x)
        else:
            current_S.add(x)
        
        # Record the 1-indexed query index (i+1) for element x
        # x is 1-indexed, so use x-1 for 0-indexed list access.
        toggle_times[x - 1].append(i + 1)

        # Store the size of S after this query (i+1)
        S_sizes[i + 1] = len(current_S)

    # --- Step 2: Compute prefix sums for S_sizes ---
    
    # prefix_sum_S_sizes[k] will store the sum S_sizes[1] + ... + S_sizes[k].
    # prefix_sum_S_sizes[0] is 0.
    prefix_sum_S_sizes = [0] * (Q + 1)
    for k in range(1, Q + 1):
        prefix_sum_S_sizes[k] = prefix_sum_S_sizes[k - 1] + S_sizes[k]

    # Helper function to get the sum of S_sizes in a range [start_q, end_q]
    # Sum = S_sizes[start_q] + ... + S_sizes[end_q]
    # This is efficiently computed using prefix sums.
    def get_sum_S_sizes_in_range(start_q, end_q):
        # If the range is invalid (start > end), sum is 0.
        if start_q > end_q:
            return 0
        # Formula: prefix_sum_S_sizes[end_q] - prefix_sum_S_sizes[start_q - 1]
        return prefix_sum_S_sizes[end_q] - prefix_sum_S_sizes[start_q - 1]

    # --- Step 3: Calculate final A values ---
    
    # Initialize the result array A with N zeros.
    A = [0] * N

    # Iterate through each element j (from 1 to N)
    # j_idx corresponds to j-1 in 0-indexed arrays.
    for j_idx in range(N):
        # Get the list of query indices where element (j_idx + 1) was toggled.
        time_list = toggle_times[j_idx]
        
        # If the element was never toggled, its A value remains 0.
        if not time_list:
            continue

        # Iterate through the toggle times in pairs.
        # Each pair (t_k, t_{k+1}) defines an interval [t_k, t_{k+1}-1]
        # during which element (j_idx+1) was in S.
        # If there's an unmatched t_k (list length is odd), the interval is [t_k, Q].
        for k in range(0, len(time_list), 2):
            start_q = time_list[k] # Query index when element was added to S
            
            if k + 1 < len(time_list):
                # Element was removed at time_list[k+1].
                # So it was in S during queries from start_q up to (time_list[k+1] - 1).
                end_q = time_list[k+1] - 1
            else:
                # Element was added at time_list[k] but not removed by query Q.
                # So it is in S from start_q until the end of all queries (Q).
                end_q = Q
            
            # Add the sum of |S| during this interval to A[j_idx].
            A[j_idx] += get_sum_S_sizes_in_range(start_q, end_q)
            
    # Print the final array A, with elements separated by spaces.
    sys.stdout.write(" ".join(map(str, A)) + "
")

# Call the solve function to run the program
solve()