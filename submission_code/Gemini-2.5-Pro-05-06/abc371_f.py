import sys

def main():
    N = int(sys.stdin.readline())
    X_str = sys.stdin.readline().split()
    # Convert X to list of ints. Using a loop for potentially faster conversion with PyPy or large N.
    X = [0] * N
    for i in range(N):
        X[i] = int(X_str[i])
        
    Q = int(sys.stdin.readline())

    # Initial Y array (0-indexed)
    # Y[i] corresponds to (i+1)-th person
    # Y[i] = X[i] - (i+1)
    Y = [0] * N
    for i in range(N):
        Y[i] = X[i] - (i + 1) 
    
    # Segment Tree stores sum, min, max for ranges of Y
    # It's 1-indexed: node 1 is root, covers Y[0...N-1]
    # Array sizes are 4*N for safety.
    tree_sum = [0] * (4 * N)
    tree_min = [0] * (4 * N)
    tree_max = [0] * (4 * N)
    # lazy[node] stores value for a pending range set operation on node's range
    # None means no pending operation
    lazy = [None] * (4 * N) 

    # Build segment tree from initial Y array
    def build(node, start, end): # node index, covers Y[start...end]
        if start == end:
            tree_sum[node] = Y[start]
            tree_min[node] = Y[start]
            tree_max[node] = Y[start]
        else:
            mid = (start + end) // 2
            build(node * 2, start, mid)
            build(node * 2 + 1, mid + 1, end)
            tree_sum[node] = tree_sum[node * 2] + tree_sum[node * 2 + 1]
            tree_min[node] = min(tree_min[node * 2], tree_min[node * 2 + 1])
            tree_max[node] = max(tree_max[node * 2], tree_max[node * 2 + 1])

    if N > 0: # Build only if there are persons
        build(1, 0, N - 1)

    # Apply lazy tag on 'node' to its children and clear lazy tag on 'node'
    def push(node, start, end):
        if lazy[node] is not None:
            val = lazy[node]
            
            # Update current node's values based on its lazy tag
            tree_sum[node] = (end - start + 1) * val
            tree_min[node] = val
            tree_max[node] = val
            
            if start != end: # If not a leaf, propagate to children
                lazy[node * 2] = val
                lazy[node * 2 + 1] = val
            lazy[node] = None # Lazy tag consumed for this node

    # Query sum of Y values in range [L, R] (inclusive, 0-indexed)
    def query_sum_range(node, start, end, L, R):
        if R < start or end < L or L > R: # Query range empty or no overlap
            return 0
        
        push(node, start, end) # Apply pending lazy updates before using/querying node

        if L <= start and end <= R: # Current segment fully within query range
            return tree_sum[node]
        
        mid = (start + end) // 2
        p1 = query_sum_range(node * 2, start, mid, L, R)
        p2 = query_sum_range(node * 2 + 1, mid + 1, end, L, R)
        return p1 + p2

    # Set Y values in range [L, R] to 'val'
    def update_range_set(node, start, end, L, R, val):
        push(node, start, end) # Apply parent's lazy tag first to current node

        if R < start or end < L or L > R: # Query range empty or no overlap
            return

        if L <= start and end <= R: # Current segment fully within update range
            lazy[node] = val # Set new lazy tag for this node
            push(node, start, end) # Apply it immediately to update this node's values
                                   # and mark children (if any) for lazy update
            return

        mid = (start + end) // 2
        update_range_set(node * 2, start, mid, L, R, val)
        update_range_set(node * 2 + 1, mid + 1, end, L, R, val)
        
        # Update node's aggregate values from its children
        tree_sum[node] = tree_sum[node * 2] + tree_sum[node * 2 + 1]
        tree_min[node] = min(tree_min[node * 2], tree_min[node * 2 + 1])
        tree_max[node] = max(tree_max[node * 2], tree_max[node * 2 + 1])

    # Find first index k in [qL, qR] such that Y[k] > V_thresh
    # Returns qR + 1 if not found (marker for not found)
    def find_first_gt(node, seg_L, seg_R, qL, qR, V_thresh):
        if seg_R < qL or seg_L > qR or qL > qR: 
            return qR + 1 

        push(node, seg_L, seg_R) 

        if tree_max[node] <= V_thresh: 
            return qR + 1

        if seg_L == seg_R: 
            return seg_L 
        
        mid = (seg_L + seg_R) // 2
        ans = qR + 1
        
        if qL <= mid : 
            ans = find_first_gt(node * 2, seg_L, mid, qL, qR, V_thresh)
        
        if ans <= qR: 
            return ans
        
        if qR > mid:
             ans = find_first_gt(node * 2 + 1, mid + 1, seg_R, qL, qR, V_thresh)
        return ans

    # Find last index k in [qL, qR] such that Y[k] < V_thresh
    # Returns qL - 1 if not found (marker for not found)
    def find_last_lt(node, seg_L, seg_R, qL, qR, V_thresh):
        if seg_R < qL or seg_L > qR or qL > qR:
            return qL - 1
        
        push(node, seg_L, seg_R)

        if tree_min[node] >= V_thresh: 
            return qL - 1
        
        if seg_L == seg_R: 
            return seg_L

        mid = (seg_L + seg_R) // 2
        ans = qL - 1
        
        if qR > mid: # Check right child first
            ans = find_last_lt(node * 2 + 1, mid + 1, seg_R, qL, qR, V_thresh)
        
        if ans >= qL: 
            return ans
        
        if qL <= mid: # Then check left child
            ans = find_last_lt(node * 2, seg_L, mid, qL, qR, V_thresh)
        return ans

    total_movements = 0

    if N == 0: # Constraints: N >= 1, so this block is defensive.
        for _ in range(Q): sys.stdin.readline() 
        sys.stdout.write("0
")
        return

    for _ in range(Q):
        T_q_str, G_q_str = sys.stdin.readline().split()
        T_q = int(T_q_str) # Person number (1-indexed)
        G_q = int(G_q_str) # Target coordinate
        
        person_idx = T_q - 1 # Convert to 0-indexed for Y array
        
        # Target Y value for Y[person_idx]. Note T_q in G_q - T_q is 1-indexed.
        V = G_q - T_q 

        # Cost for person_idx changing its Y value
        # Query Y[person_idx]'s old value. Sum query for range [person_idx, person_idx].
        Y_old_val_person_idx = query_sum_range(1, 0, N - 1, person_idx, person_idx)
        total_movements += abs(V - Y_old_val_person_idx)
        
        # Update Y[person_idx] to V. This is a point update.
        update_range_set(1, 0, N - 1, person_idx, person_idx, V)

        # Left side: indices < person_idx. Range is [0, person_idx-1]
        # Find L_boundary = first k in [0, person_idx-1] such that Y[k] > V
        if person_idx > 0: # Only if there's a left side
            # query_L=0, query_R=person_idx-1
            L_boundary = find_first_gt(1, 0, N - 1, 0, person_idx - 1, V)
            # L_boundary is an index. If not found, it's (person_idx-1) + 1 = person_idx.
            # Range to update: [L_boundary, person_idx-1]
            if L_boundary <= person_idx - 1: # If a valid range exists
                sum_old_Y_left = query_sum_range(1, 0, N - 1, L_boundary, person_idx - 1)
                count_left = (person_idx - 1) - L_boundary + 1
                total_movements += sum_old_Y_left - count_left * V # Y_old > V, so Y_old - V
                update_range_set(1, 0, N - 1, L_boundary, person_idx - 1, V)

        # Right side: indices > person_idx. Range is [person_idx+1, N-1]
        # Find R_boundary = last k in [person_idx+1, N-1] such that Y[k] < V
        if person_idx < N - 1: # Only if there's a right side
            # query_L=person_idx+1, query_R=N-1
            R_boundary = find_last_lt(1, 0, N - 1, person_idx + 1, N - 1, V)
            # R_boundary is an index. If not found, it's (person_idx+1) - 1 = person_idx.
            # Range to update: [person_idx+1, R_boundary]
            if R_boundary >= person_idx + 1: # If a valid range exists
                sum_old_Y_right = query_sum_range(1, 0, N - 1, person_idx + 1, R_boundary)
                count_right = R_boundary - (person_idx + 1) + 1
                total_movements += count_right * V - sum_old_Y_right # Y_old < V, so V - Y_old
                update_range_set(1, 0, N - 1, person_idx + 1, R_boundary, V)
                
    sys.stdout.write(str(total_movements) + "
")

if __name__ == '__main__':
    main()