# YOUR CODE HERE
import sys

# Increase recursion depth for deep segment tree operations if needed
# Set a reasonable limit based on constraints N <= 2*10^5 -> Log2(N) approx 18. 
# Python's default limit is usually 1000, which is sufficient.
# sys.setrecursionlimit(200010) 

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    Q = int(sys.stdin.readline())

    # Precompute P_j: largest index k such that 1 <= k < j and A[k-1] <= A[j-1]/2
    # P is 1-indexed array storing the 1-based index k.
    P = [0] * (N + 1)
    # Use two pointers technique for efficient computation in O(N)
    ptr_i = 0 # ptr_i is 0-indexed, points to A[ptr_i], represents candidate index k = ptr_i + 1
    for j_idx in range(1, N): # j_idx is 0-indexed, points to A[j_idx], represents index j = j_idx + 1
        # Advance ptr_i while the condition A[ptr_i] * 2 <= A[j_idx] holds
        # and ensure ptr_i corresponds to an index k < j (i.e., ptr_i < j_idx)
        while ptr_i < j_idx and A[ptr_i] * 2 <= A[j_idx]:
             ptr_i += 1
        
        # After the loop, ptr_i points to the first index k such that A[k-1]*2 > A[j_idx]
        # The largest index k satisfying the condition A[k-1]*2 <= A[j_idx] is ptr_i (as a 1-based index)
        # Note: If A[0]*2 > A[j_idx], ptr_i remains 0.
        # P[j] stores this largest k (1-based index). If no such k exists, P[j] remains 0.
        P[j_idx + 1] = ptr_i 

    # Segment tree structure for Range Maximum Query on available indices
    # Stores the maximum available index in the represented range. 0 indicates no available index.
    tree = [0] * (4 * N)
    # Global tracker of availability status for indices between queries.
    # True if index is available, False otherwise.
    available_global = [False] * (N + 1) 

    # Build function initializes the segment tree. All nodes store 0 initially.
    def build(v, tl, tr): 
        if tl == tr:
            tree[v] = 0 
        else:
            tm = (tl + tr) // 2
            build(2 * v, tl, tm)
            build(2 * v + 1, tm + 1, tr)
            tree[v] = 0 

    # Update function sets the value for a position 'pos' in the segment tree.
    # Used to mark an index as available (value = index) or unavailable (value = 0).
    def update(v, tl, tr, pos, new_val):
        if tl == tr:
            tree[v] = new_val
        else:
            tm = (tl + tr) // 2
            if pos <= tm:
                update(2 * v, tl, tm, pos, new_val)
            else:
                update(2 * v + 1, tm + 1, tr, pos, new_val)
            # Internal node stores maximum of its children
            tree[v] = max(tree[2 * v], tree[2 * v + 1])

    # Range Maximum Query function. Returns the maximum value (index) in the range [l, r].
    def query_fixed(v, tl, tr, l, r):
        # If query range is empty or does not overlap node range, return 0
        if l > r or tl > r or tr < l: 
            return 0 
        # If node range is fully contained within query range
        if l <= tl and tr <= r: 
            return tree[v]
        
        tm = (tl + tr) // 2
        max_val = 0
        # Recursively query left child if overlap exists
        max_val = max(max_val, query_fixed(2 * v, tl, tm, l, r))
        # Recursively query right child if overlap exists
        max_val = max(max_val, query_fixed(2 * v + 1, tm + 1, tr, l, r))
        
        return max_val

    
    # Initial build of the segment tree
    build(1, 1, N)
    
    # Store results to print all at once at the end
    query_results = []

    for _ in range(Q):
        L, R = map(int, sys.stdin.readline().split())
        
        # List to store changes made during this query to revert them later
        # Stores tuples (position, original_value_before_change)
        current_query_changes = []

        # Activate indices in the query range [L, R] if they are currently inactive
        for k in range(L, R+1):
            if not available_global[k]: # Check global state
                update(1, 1, N, k, k) # Set tree leaf value to k (activate)
                available_global[k] = True # Update global state
                current_query_changes.append((k, 0)) # Record: index k activated from state 0

        k_count = 0 # Counter for pairs formed in this query
        
        # Implement the greedy strategy: Iterate j from R down to L
        for j in range(R, L - 1, -1):
            # Check if index j is available
            if available_global[j]: 
                
                # Find the maximum possible index k for a top piece for j.
                k_max = P[j] 
                # The actual upper bound for search is min(j-1, k_max)
                query_upper_bound = min(j - 1, k_max)
                
                # Check if there's any potential top piece index within the query range [L, R]
                if query_upper_bound >= L:
                    
                    # Find the largest available index i in range [L, query_upper_bound]
                    found_i = query_fixed(1, 1, N, L, query_upper_bound)
                    
                    # If a valid available index i is found (found_i >= L ensures it's within L..R and available)
                    if found_i >= L: 
                        k_count += 1 # Increment pair count
                        
                        # Mark index i as unavailable
                        update(1, 1, N, found_i, 0) # Set tree leaf value to 0
                        available_global[found_i] = False # Update global state
                        current_query_changes.append((found_i, found_i)) # Record: index i deactivated from state i

                        # Mark index j as unavailable
                        update(1, 1, N, j, 0) # Set tree leaf value to 0
                        available_global[j] = False # Update global state
                        current_query_changes.append((j, j)) # Record: index j deactivated from state j

                    else:
                         # No suitable available index i found in the range.
                         # Mark j as unavailable (cannot be used as a bottom piece).
                         update(1, 1, N, j, 0)
                         available_global[j] = False
                         current_query_changes.append((j, j)) # Record: index j deactivated from state j
                else:
                     # k_max < L or j-1 < L means no possible index i in range [L, R].
                     # Mark j as unavailable.
                     update(1, 1, N, j, 0)
                     available_global[j] = False
                     current_query_changes.append((j, j)) # Record: index j deactivated from state j

        # Append the result for the current query
        query_results.append(k_count)

        # Revert changes made during this query to restore the state for the next query
        for pos, original_state_val in reversed(current_query_changes):
             update(1, 1, N, pos, original_state_val) # Restore value in segment tree
             available_global[pos] = (original_state_val > 0) # Restore global availability state
             

    # Print all results
    for res in query_results:
        print(res)

# Run the main solver function
solve()