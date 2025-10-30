import sys

# It's good practice to set a higher recursion limit for segment tree operations if N is large.
# Default recursion limit in Python is often 1000 or 3000.
# For N=5e5, segment tree depth is log2(N) ~ 19. Max path in query recursion is ~19.
# So default limit should be fine for queries. Build is iterative.
# sys.setrecursionlimit(10**6) # Might be needed for extremely deep recursions or specific environments

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Prefix sums P: P[k] = sum(A[0]...A[k-1])
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] + A[i]

    # V_L[i] = A[i] + P[i+1] (where P[i+1] = sum(A[0]...A[i]))
    # Barrier condition for left expansion: V_L[idx] >= P[r_curr+1]
    V_L = [(A[i] + P[i+1]) for i in range(N)]

    # V_R[i] = A[i] - P[i] (where P[i] = sum(A[0]...A[i-1]))
    # Barrier condition for right expansion: V_R[idx] >= -P[l_curr]
    V_R = [(A[i] - P[i]) for i in range(N)]
    
    st_n = 1
    while st_n < N:
        st_n *= 2
    
    # Segment Tree for V_L: stores max(V_L[i]) in range
    # Used to find rightmost index idx in [q_low, q_high] s.t. V_L[idx] >= threshold
    st_vl_val = [-float('inf')] * (2 * st_n)
    if N > 0 : # Check N>0 for st_n potentially being 0 if N=0 (not per constraints)
        for i in range(N):
            st_vl_val[i + st_n] = V_L[i]
        # Fill unused leaf spots if N is not power of 2
        # These are outside the [0, N-1] range so should not be returned by queries matching V_L[idx]
        for i in range(N, st_n): 
            st_vl_val[i + st_n] = -float('inf') 
        # Build tree
        for i in range(st_n - 1, 0, -1):
            st_vl_val[i] = max(st_vl_val[i * 2], st_vl_val[i * 2 + 1])

    def query_vl(node_idx, current_l, current_r, query_l, query_r, threshold):
        if query_l > query_r: 
            return -1 
        # If current segment [current_l, current_r] does not overlap with [query_l, query_r]
        # or max value in current segment is less than threshold
        if current_l > query_r or current_r < query_l or st_vl_val[node_idx] < threshold:
            return -1
        
        if current_l == current_r: # Leaf node
            return current_l

        mid = (current_l + current_r) // 2
        
        # Try right child first for rightmost result
        res_r = query_vl(node_idx * 2 + 1, mid + 1, current_r, query_l, query_r, threshold)
        if res_r != -1:
            return res_r
        
        return query_vl(node_idx * 2, current_l, mid, query_l, query_r, threshold)

    # Segment Tree for V_R: stores max(V_R[i]) in range
    # Used to find leftmost index idx in [q_low, q_high] s.t. V_R[idx] >= threshold
    st_vr_val = [-float('inf')] * (2 * st_n)
    if N > 0:
        for i in range(N):
            st_vr_val[i + st_n] = V_R[i]
        for i in range(N, st_n):
            st_vr_val[i + st_n] = -float('inf')
        for i in range(st_n - 1, 0, -1):
            st_vr_val[i] = max(st_vr_val[i * 2], st_vr_val[i * 2 + 1])
    
    def query_vr(node_idx, current_l, current_r, query_l, query_r, threshold):
        if query_l > query_r:
            return N # Using N as "not found" indicator for right search
        if current_l > query_r or current_r < query_l or st_vr_val[node_idx] < threshold:
            return N
        
        if current_l == current_r: # Leaf node
            return current_l

        mid = (current_l + current_r) // 2
        
        # Try left child first for leftmost result
        res_l = query_vr(node_idx * 2, current_l, mid, query_l, query_r, threshold)
        if res_l != N:
            return res_l
        
        return query_vr(node_idx * 2 + 1, mid + 1, current_r, query_l, query_r, threshold)
    
    results = []
    for k_idx in range(N):
        l_curr, r_curr = k_idx, k_idx
        
        while True:
            prev_l, prev_r = l_curr, r_curr
            
            # Expand left
            if l_curr > 0:
                target_vl = P[r_curr+1]
                barrier_idx_l = query_vl(1, 0, st_n-1, 0, l_curr-1, target_vl) # query range is [0, st_n-1] for tree
                l_curr = barrier_idx_l + 1
            
            # Expand right
            if r_curr < N-1:
                target_vr = -P[l_curr]
                barrier_idx_r = query_vr(1, 0, st_n-1, r_curr+1, N-1, target_vr)
                r_curr = barrier_idx_r - 1
            
            if l_curr == prev_l and r_curr == prev_r:
                break
        
        results.append(P[r_curr+1] - P[l_curr])
        
    sys.stdout.write(" ".join(map(str, results)) + "
")

solve()