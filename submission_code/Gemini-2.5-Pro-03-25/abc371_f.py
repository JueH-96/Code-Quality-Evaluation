# YOUR CODE HERE
import sys

# Set higher recursion depth for safety, although segment tree operations 
# typically have logarithmic depth. Python's default limit might be too low for deep recursion.
# This might be needed for deep chains of updates/queries or large N.
try:
    # Set a large recursion depth, e.g., N + Q + buffer
    sys.setrecursionlimit(400010 + 10) 
except Exception as e: # Handle potential OS limits or restrictions
    # Optional: print a warning if setting recursion depth fails
    # print(f"Could not set recursion depth: {e}", file=sys.stderr)
    pass 


def solve():
    N = int(sys.stdin.readline())
    X = list(map(int, sys.stdin.readline().split()))
    
    Q = int(sys.stdin.readline())
    
    # Using 1-based indexing for persons P[1..N] for easier problem mapping
    P = [0] * (N + 1) 
    for i in range(N):
        P[i+1] = X[i]

    # Using 1-based indexing for gaps E[1..N-1]. Store excess space.
    # E[k] = P[k+1] - P[k] - 1 is the free space between person k and k+1.
    # Minimum separation is 1, so E[k] >= 0.
    # E array has size N, E[0] is unused, indices 1..N-1 are valid gap indices.
    E = [0] * N 
    for k in range(1, N):
        # Ensure gaps are non-negative. They represent excess space beyond minimum 1 unit.
        E[k] = max(0, P[k+1] - P[k] - 1) 

    # Segment tree setup: Covers indices 1..N-1 (the gaps).
    # Calculate the smallest power of 2 >= N-1 for tree base size.
    tree_base_size = N - 1
    if tree_base_size <= 0: # Handle N=1 case where there are no gaps
        tree_base_size = 1 # Use minimum base size 1 for tree construction logic simplicity

    tree_size = 1
    while tree_size < tree_base_size:
        tree_size *= 2
    
    # Segment tree arrays: Stores sum of E_k and sum of E_k * k
    tree_sum_E = [0] * (2 * tree_size)
    tree_sum_Ek = [0] * (2 * tree_size)
    # Lazy propagation flag for range zero updates
    lazy_zero = [False] * (2 * tree_size)

    # Segment Tree Build function
    def build(k, L, R):
        # Builds segment tree recursively for indices 1..tree_size
        # Actual gap data E[k] exists only for indices 1..N-1
        
        # If the current node's range [L,R] is entirely outside valid gap indices [1, N-1], stop.
        if L > N - 1 and L <= R: 
             return 

        if L == R: # Leaf node
            # Check if L is a valid gap index before accessing E[L]
            if 1 <= L <= N-1: 
                 tree_sum_E[k] = E[L]
                 tree_sum_Ek[k] = E[L] * L
            # else: it's a padding node (L > N-1 or L<1), values remain 0.
            return

        mid = (L + R) // 2
        build(2 * k, L, mid)
        build(2 * k + 1, mid + 1, R)
        
        # Combine results from children after they are built
        tree_sum_E[k] = tree_sum_E[2 * k] + tree_sum_E[2 * k + 1]
        tree_sum_Ek[k] = tree_sum_Ek[2 * k] + tree_sum_Ek[2 * k + 1]

    # Build the segment tree only if N > 1 (at least one gap exists)
    if N > 1: 
        build(1, 1, tree_size)

    # Helper function: Push lazy flag down to children
    def push(k, L, R):
        if lazy_zero[k]:
            # Apply the zeroing effect to the current node
            tree_sum_E[k] = 0
            tree_sum_Ek[k] = 0
            # If not a leaf node, mark children as lazy
            if L != R: 
                lazy_zero[2*k] = True
                lazy_zero[2*k+1] = True
            # Reset lazy flag for the current node
            lazy_zero[k] = False 

    # Update function: Set range [queryL, queryR] to zero
    def range_update_zero(k, L, R, queryL, queryR):
        push(k, L, R) # Process lazy flag for current node first
        
        # If current node range [L,R] is completely outside query range [queryL, queryR]
        if R < queryL or L > queryR: 
            return
        
        # If current node range is fully contained within query range
        if queryL <= L and R <= queryR: 
            lazy_zero[k] = True # Mark node k as lazy zeroed
            push(k, L, R) # Apply the effect and potentially propagate
            return

        # Otherwise, partial overlap: recurse down to children
        mid = (L + R) // 2
        range_update_zero(2 * k, L, mid, queryL, queryR)
        range_update_zero(2 * k + 1, mid + 1, R, queryL, queryR)
        
        # Update current node's sums based on potentially modified children
        tree_sum_E[k] = tree_sum_E[2 * k] + tree_sum_E[2 * k + 1]
        tree_sum_Ek[k] = tree_sum_Ek[2 * k] + tree_sum_Ek[2 * k + 1]

    # Update function: Point update (add delta_E to E[target_idx])
    def point_update(k, L, R, target_idx, delta_E):
        push(k, L, R) # Process lazy flag first
        
        # If target index is outside current node range
        if R < target_idx or L > target_idx: 
            return

        # If leaf node corresponding to the target index
        if L == R: 
             tree_sum_E[k] += delta_E
             tree_sum_Ek[k] += delta_E * L # L == target_idx
             return

        # Otherwise, recurse down
        mid = (L + R) // 2
        if target_idx <= mid:
             # Update left child, then ensure right child's lazy flag is processed before combining
             point_update(2 * k, L, mid, target_idx, delta_E)
             push(2 * k + 1, mid + 1, R) 
        else:
             # Update right child, then ensure left child's lazy flag is processed before combining
             point_update(2 * k + 1, mid + 1, R, target_idx, delta_E)
             push(2 * k, L, mid) 
         
        # Update current node's sums from children
        tree_sum_E[k] = tree_sum_E[2 * k] + tree_sum_E[2 * k + 1]
        tree_sum_Ek[k] = tree_sum_Ek[2 * k] + tree_sum_Ek[2 * k + 1]

    # Query function: Get sums (sum_E, sum_Ek) in range [queryL, queryR]
    def query(k, L, R, queryL, queryR):
        # Ensure query range is valid to avoid infinite recursion or incorrect results
        if queryL > queryR: return (0,0)

        push(k, L, R) # Process lazy flag first
        
        # If current node range is outside query range
        if R < queryL or L > queryR: 
            return (0, 0) 
        
        # If current node range is fully contained within query range
        if queryL <= L and R <= queryR: 
             return (tree_sum_E[k], tree_sum_Ek[k])

        # Partial overlap: recurse down and combine results
        mid = (L + R) // 2
        resL = query(2 * k, L, mid, queryL, queryR)
        resR = query(2 * k + 1, mid + 1, R, queryL, queryR)
        return (resL[0] + resR[0], resL[1] + resR[1])

    # Helper functions to find the index K using binary search on segment tree queries
    # Find smallest K in [Ti, N-1] s.t. sum E_k from Ti to K >= delta
    def find_K_east(Ti, delta):
        best_K = N # Default means need space from E_N (infinity)
        search_low = Ti
        search_high = N - 1

        while search_low <= search_high:
            mid = (search_low + search_high) // 2
            # Query sum E in range [Ti, mid]
            sum_E_range, _ = query(1, 1, tree_size, Ti, mid)
            
            if sum_E_range >= delta:
                 best_K = mid # Found a potential K, try for smaller K
                 search_high = mid - 1 
            else:
                 # Need more sum, search in larger indices
                 search_low = mid + 1 
        return best_K

    # Find largest K in [1, Ti-1] s.t. sum E_k from K to Ti-1 >= delta
    def find_K_west(Ti, delta):
        best_K = 0 # Default means need space from E_0 (infinity)
        search_low = 1
        search_high = Ti - 1

        while search_low <= search_high:
             mid = (search_low + search_high) // 2
             # Query sum E in range [mid, Ti-1]
             sum_E_range, _ = query(1, 1, tree_size, mid, Ti - 1)
             
             if sum_E_range >= delta:
                 best_K = mid # Found a potential K, try for larger K (smaller index)
                 search_low = mid + 1 
             else:
                 # Need more sum, search in smaller indices (which corresponds to larger range start 'mid')
                 search_high = mid - 1 
        return best_K

    total_cost = 0

    # Main loop over queries
    for _ in range(Q):
        line = sys.stdin.readline().split()
        Ti = int(line[0])
        Gi = int(line[1])

        delta = Gi - P[Ti] # Required displacement

        if delta == 0: # No movement needed
            continue

        current_task_cost = 0

        if delta > 0: # Move East by delta
             # Handle edge cases N=1 and Ti=N
             if N == 1: 
                current_task_cost = delta
             elif Ti == N: # Rightmost person moves east freely
                 current_task_cost = delta
                 # Update the gap E[N-1] if N > 1
                 if N > 1: 
                     point_update(1, 1, tree_size, N - 1, delta)
             
             # General case: Ti < N
             else: 
                 # Check total available space in gaps E[Ti..N-1]
                 S_avail, _ = query(1, 1, tree_size, Ti, N - 1)
                 
                 if S_avail >= delta: # Enough space available within existing gaps
                     # Find the critical gap index K
                     K = find_K_east(Ti, delta)
                     
                     # Calculate cost using derived formula
                     # Cost = sum(u_k * k for k=Ti..K) - (Ti-1)*delta
                     sum_E_Ti_Kminus1, sum_Ek_Ti_Kminus1 = query(1, 1, tree_size, Ti, K - 1)
                     u_K = delta - sum_E_Ti_Kminus1 # Amount needed from gap K
                     current_task_cost = sum_Ek_Ti_Kminus1 + u_K * K - (Ti - 1) * delta

                     # Update the segment tree: zero out used ranges, update partially used K, update created gap Ti-1
                     if Ti <= K-1: # Zero out fully consumed gaps
                        range_update_zero(1, 1, tree_size, Ti, K - 1)
                     if K <= N-1: # Update the partially consumed gap K
                        point_update(1, 1, tree_size, K, -u_K)
                     if Ti > 1: # Update the gap created to the left
                         point_update(1, 1, tree_size, Ti - 1, delta)

                 else: # S_avail < delta: Need space from infinity (E_N)
                     delta_prime = delta - S_avail # Amount needed from infinity
                     
                     # Calculate cost using formula for using infinite space
                     # Cost = sum(E_k * k for k=Ti..N-1) + delta_prime * N - (Ti-1)*delta
                     _, sum_Ek_Ti_Nminus1 = query(1, 1, tree_size, Ti, N - 1)
                     current_task_cost = sum_Ek_Ti_Nminus1 + delta_prime * N - (Ti - 1) * delta

                     # Update the segment tree: zero out all used gaps, update created gap Ti-1
                     if Ti <= N-1: # Zero out all consumed gaps up to N-1
                         range_update_zero(1, 1, tree_size, Ti, N - 1)
                     if Ti > 1: # Update the gap created to the left
                         point_update(1, 1, tree_size, Ti - 1, delta)

        else: # Move West by delta_abs = abs(delta)
             delta_abs = -delta
             # Handle edge cases N=1 and Ti=1
             if N == 1: 
                 current_task_cost = delta_abs
             elif Ti == 1: # Leftmost person moves west freely
                 current_task_cost = delta_abs
                 # Update the gap E[1] if N > 1
                 if N > 1: 
                     point_update(1, 1, tree_size, 1, delta_abs)
            
             # General case: Ti > 1
             else: 
                 # Check total available space in gaps E[1..Ti-1]
                 S_avail, _ = query(1, 1, tree_size, 1, Ti - 1)

                 if S_avail >= delta_abs: # Enough space available within existing gaps
                     # Find the critical gap index K
                     K = find_K_west(Ti, delta_abs)
                     
                     # Calculate cost using derived formula
                     # Cost = Ti * delta_abs - (sum(E_k * k for k=K+1..Ti-1) + u_K * K)
                     sum_E_Kplus1_Timinus1, sum_Ek_Kplus1_Timinus1 = query(1, 1, tree_size, K + 1, Ti - 1)
                     u_K = delta_abs - sum_E_Kplus1_Timinus1 # Amount needed from gap K
                     current_task_cost = Ti * delta_abs - (sum_Ek_Kplus1_Timinus1 + u_K * K)

                     # Update the segment tree: zero out used ranges, update partially used K, update created gap Ti
                     if K+1 <= Ti-1: # Zero out fully consumed gaps
                        range_update_zero(1, 1, tree_size, K + 1, Ti - 1)
                     if K >= 1: # Update the partially consumed gap K
                        point_update(1, 1, tree_size, K, -u_K)
                     if Ti <= N-1: # Update the gap created to the right (gap Ti)
                        point_update(1, 1, tree_size, Ti, delta_abs)

                 else: # S_avail < delta_abs: Need space from infinity (E_0)
                     delta_prime = delta_abs - S_avail # Amount needed from infinity
                     
                     # Calculate cost using formula for using infinite space
                     # Cost = Ti * S_avail - sum(E_k*k for k=1..Ti-1) + delta_prime * Ti
                     _, sum_Ek_1_Timinus1 = query(1, 1, tree_size, 1, Ti - 1)
                     current_task_cost = Ti * S_avail - sum_Ek_1_Timinus1 + delta_prime * Ti

                     # Update the segment tree: zero out all used gaps, update created gap Ti
                     if 1 <= Ti-1: # Zero out all consumed gaps from 1 to Ti-1
                        range_update_zero(1, 1, tree_size, 1, Ti - 1)
                     if Ti <= N-1: # Update the gap created to the right
                         point_update(1, 1, tree_size, Ti, delta_abs)

        total_cost += current_task_cost
        # Update the position of the person after movement
        P[Ti] = Gi 

    print(total_cost)

solve()