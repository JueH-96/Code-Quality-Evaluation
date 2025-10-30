import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Precompute prefix sums for efficient range sum queries
    # prefix_sum[i] will store sum A[0]...A[i-1]
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + A[i]

    # Segment Tree for Range Maximum Query (RMQ)
    # Each node stores (max_val, index_of_max_val) in its range.
    
    # tree array uses 1-based indexing for convenience in calculations
    # Indices 0 to N-1 for A, so tree operates on ranges based on these indices.
    tree = [(0, -1)] * (4 * N) # (max_value, index_of_max_value)

    # Builds the segment tree
    # node_idx: current node in the tree array
    # start_range, end_range: current segment's boundaries in the A array
    def build(node_idx, start_range, end_range):
        if start_range == end_range:
            tree[node_idx] = (A[start_range], start_range)
        else:
            mid = (start_range + end_range) // 2
            build(2 * node_idx, start_range, mid)
            build(2 * node_idx + 1, mid + 1, end_range)
            # Parent node stores max of its children
            tree[node_idx] = max(tree[2 * node_idx], tree[2 * node_idx + 1])

    # Finds the leftmost index in [query_L, query_R] (inclusive) where A[idx] >= value.
    # Returns N if no such element is found in the specified range.
    def find_first_ge_left(node_idx, current_start, current_end, query_L, query_R, value):
        # Current segment is entirely outside the query range OR
        # The maximum value in the current segment is less than 'value', so no element >= 'value' exists here.
        if query_R < current_start or current_end < query_L or tree[node_idx][0] < value:
            return N # Indicate not found
        
        # If current segment is a leaf node, check its value directly
        if current_start == current_end:
            return current_start # It must be >= value from the check above

        mid = (current_start + current_end) // 2
        
        # Try to find in the left child first (because we want the leftmost index)
        res = find_first_ge_left(2 * node_idx, current_start, mid, query_L, query_R, value)
        if res != N: # If found in left child, return it
            return res
        
        # If not found in left child, search in the right child
        return find_first_ge_left(2 * node_idx + 1, mid + 1, current_end, query_L, query_R, value)

    # Finds the rightmost index in [query_L, query_R] (inclusive) where A[idx] >= value.
    # Returns -1 if no such element is found in the specified range.
    def find_first_ge_right(node_idx, current_start, current_end, query_L, query_R, value):
        # Current segment is entirely outside the query range OR
        # The maximum value in the current segment is less than 'value', so no element >= 'value' exists here.
        if query_R < current_start or current_end < query_L or tree[node_idx][0] < value:
            return -1 # Indicate not found
        
        # If current segment is a leaf node, check its value directly
        if current_start == current_end:
            return current_start # It must be >= value from the check above

        mid = (current_start + current_end) // 2
        
        # Try to find in the right child first (because we want the rightmost index)
        res = find_first_ge_right(2 * node_idx + 1, mid + 1, current_end, query_L, query_R, value)
        if res != -1: # If found in right child, return it
            return res
        
        # If not found in right child, search in the left child
        return find_first_ge_right(2 * node_idx, current_start, mid, query_L, query_R, value)

    # Build the segment tree for the entire array A
    build(1, 0, N-1)

    ans = [0] * N # Array to store results for each K

    # Iterate for each starting slime K
    for k in range(N):
        l_curr = k # Current leftmost index of Takahashi's absorbed range
        r_curr = k # Current rightmost index of Takahashi's absorbed range
        current_sum = A[k] # Current total size of Takahashi

        changed = True # Flag to check if expansion occurred in an iteration
        while changed:
            changed = False # Assume no expansion initially for this iteration
            
            # Find the leftmost slime to the right of r_curr that is >= current_sum.
            # This slime acts as a "wall" for rightward expansion.
            # Search range for this wall is [r_curr + 1, N - 1].
            right_barrier_idx = N # Default if no wall found (can expand to N-1)
            if r_curr < N - 1: # Only search if there are elements to the right
                right_barrier_idx = find_first_ge_left(1, 0, N-1, r_curr + 1, N - 1, current_sum)
            
            # Find the rightmost slime to the left of l_curr that is >= current_sum.
            # This slime acts as a "wall" for leftward expansion.
            # Search range for this wall is [0, l_curr - 1].
            left_barrier_idx = -1 # Default if no wall found (can expand to 0)
            if l_curr > 0: # Only search if there are elements to the left
                left_barrier_idx = find_first_ge_right(1, 0, N-1, 0, l_curr - 1, current_sum)
            
            # Calculate the new potential boundaries after absorbing all possible slimes
            # up to the identified barriers.
            # Elements from new_l_potential to l_curr-1 will be absorbed.
            new_l_potential = left_barrier_idx + 1
            # Elements from r_curr+1 to new_r_potential will be absorbed.
            new_r_potential = right_barrier_idx - 1

            # Check if any expansion actually occurred in this iteration.
            # Expansion occurs if new_l_potential moved further left or new_r_potential moved further right.
            if new_l_potential < l_curr or new_r_potential > r_curr:
                # Add the sum of newly absorbed slimes to current_sum.
                # Sum of elements from new_l_potential to l_curr-1:
                current_sum += (prefix_sum[l_curr] - prefix_sum[new_l_potential])
                # Sum of elements from r_curr+1 to new_r_potential:
                current_sum += (prefix_sum[new_r_potential+1] - prefix_sum[r_curr+1])
                
                # Update current boundaries
                l_curr = new_l_potential
                r_curr = new_r_potential
                changed = True # Set flag to continue the while loop
            
        ans[k] = current_sum # Store the final maximum size for slime K

    # Print all answers separated by spaces
    print(*(ans))

# Call the solve function to run the program
solve()