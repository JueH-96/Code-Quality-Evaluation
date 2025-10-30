import sys

# Use sys.stdin.readline for faster input
# Use sys.stdout.write for faster output

def solve():
    N, Q = map(int, sys.stdin.readline().split())

    # Segment Tree implementation
    # Each node stores [color, is_uniform, lazy_color]
    # color: If is_uniform is True, this is the color for the whole range. Otherwise, 0 (mixed).
    # is_uniform: True if all elements in node's range have the same color.
    # lazy_color: 0 if no lazy update, otherwise the color to propagate.
    # Colors are 1 to N, so 0 is a safe sentinel for mixed/no lazy.
    seg_tree = [[0, False, 0] for _ in range(4 * N)]
    
    # Store counts of each color (1-indexed)
    color_counts = [0] * (N + 1)

    # Initial build of the segment tree
    # Each cell i initially has color i
    def build(node_idx, lo, hi):
        if lo == hi:
            seg_tree[node_idx] = [lo, True, 0] # Cell `lo` has color `lo`
            color_counts[lo] += 1
            return
        
        mid = (lo + hi) // 2
        build(2 * node_idx, lo, mid)
        build(2 * node_idx + 1, mid + 1, hi)
        
        # Initially, children have different colors, so this node is not uniform
        seg_tree[node_idx] = [0, False, 0] 

    # Propagate lazy updates down the tree
    def push(node_idx):
        if seg_tree[node_idx][2] != 0: # If there's a lazy color to propagate
            c_val = seg_tree[node_idx][2]
            
            # Apply lazy color to left child
            seg_tree[2 * node_idx][0] = c_val
            seg_tree[2 * node_idx][1] = True
            seg_tree[2 * node_idx][2] = c_val # Propagate lazy down
            
            # Apply lazy color to right child
            seg_tree[2 * node_idx + 1][0] = c_val
            seg_tree[2 * node_idx + 1][1] = True
            seg_tree[2 * node_idx + 1][2] = c_val # Propagate lazy down
            
            # Clear lazy for current node
            seg_tree[node_idx][2] = 0

    # Range update: set range [q_lo, q_hi] to color c_new
    def range_update(node_idx, lo, hi, q_lo, q_hi, c_new):
        # If current node's range is fully within the query range, update it
        if q_lo <= lo and hi <= q_hi:
            seg_tree[node_idx][0] = c_new
            seg_tree[node_idx][1] = True
            seg_tree[node_idx][2] = c_new # Mark for lazy propagation
            return
        
        # If current node's range is outside query range, do nothing
        if hi < q_lo or lo > q_hi:
            return

        push(node_idx) # Propagate lazy updates before recursing

        mid = (lo + hi) // 2
        range_update(2 * node_idx, lo, mid, q_lo, q_hi, c_new)
        range_update(2 * node_idx + 1, mid + 1, hi, q_lo, q_hi, c_new)

        # After children are updated, re-evaluate current node's state
        left_child = seg_tree[2 * node_idx]
        right_child = seg_tree[2 * node_idx + 1]

        # If both children are uniform and have the same color, current node is also uniform
        if left_child[1] and right_child[1] and left_child[0] == right_child[0]:
            seg_tree[node_idx][0] = left_child[0]
            seg_tree[node_idx][1] = True
        else: # Otherwise, current node is mixed
            seg_tree[node_idx][0] = 0 # Mixed color
            seg_tree[node_idx][1] = False

    # Get color of a single cell at query_pos
    def get_color(node_idx, lo, hi, query_pos):
        # If current node is uniform, its color is the answer (after lazy propagation)
        if seg_tree[node_idx][1]: 
            return seg_tree[node_idx][0]
        
        # If at a leaf node (should be uniform, but safeguard)
        if lo == hi:
             return seg_tree[node_idx][0] 
        
        push(node_idx) # Propagate lazy updates before recursing

        mid = (lo + hi) // 2
        if query_pos <= mid:
            return get_color(2 * node_idx, lo, mid, query_pos)
        else:
            return get_color(2 * node_idx + 1, mid + 1, hi, query_pos)

    # Initialize segment tree
    build(1, 1, N) 

    # Process queries
    for _ in range(Q):
        query = list(map(int, sys.stdin.readline().split()))
        q_type = query[0]

        if q_type == 1: # Type 1 query: 1 x c_new
            x, c_new = query[1], query[2]

            old_color = get_color(1, 1, N, x)

            # Find L_found (left boundary of the segment with old_color)
            L_found = x
            low, high = 1, x
            while low <= high:
                mid = (low + high) // 2
                if get_color(1, 1, N, mid) == old_color:
                    L_found = mid
                    high = mid - 1 # Try to find an even smaller index
                else:
                    low = mid + 1 # This part is not old_color, so search higher
            
            # Find R_found (right boundary of the segment with old_color)
            R_found = x
            low, high = x, N
            while low <= high:
                mid = (low + high) // 2
                if get_color(1, 1, N, mid) == old_color:
                    R_found = mid
                    low = mid + 1 # Try to find an even larger index
                else:
                    high = mid - 1 # This part is not old_color, so search lower
            
            # Update color counts
            cells_to_repaint_count = R_found - L_found + 1
            color_counts[old_color] -= cells_to_repaint_count
            color_counts[c_new] += cells_to_repaint_count

            # Perform range update on the segment tree
            range_update(1, 1, N, L_found, R_found, c_new)

        elif q_type == 2: # Type 2 query: 2 c
            c = query[1]
            sys.stdout.write(str(color_counts[c]) + "
")

solve()