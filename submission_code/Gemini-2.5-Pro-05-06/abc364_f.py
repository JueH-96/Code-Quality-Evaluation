import sys

# Increase recursion limit for potentially deep DSU find calls and segment tree recursion
sys.setrecursionlimit(4 * 10**5) 

def solve():
    N, Q = map(int, sys.stdin.readline().split())

    queries_input = []
    for i in range(Q):
        l, r, c = map(int, sys.stdin.readline().split())
        # Store original 0-indexed ID to map to query vertices N+1 ... N+Q
        queries_input.append({'L': l, 'R': r, 'C': c, 'orig_idx': i})

    queries_input.sort(key=lambda q: q['C'])

    # DSU structure for vertices 1 to N+Q
    parent = list(range(N + Q + 1))
    sz = [1] * (N + Q + 1)
    num_components = N + Q # Initially N+Q components

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def unite(i, j):
        nonlocal num_components
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            if sz[root_i] < sz[root_j]: # Union by size
                root_i, root_j = root_j, root_i
            parent[root_j] = root_i
            sz[root_i] += sz[root_j]
            num_components -= 1
            return True # Returns true if a merge happened
        return False # False if already in same component

    # Segment Tree for base vertices 1 to N
    # id_val[node_idx] stores a representative vertex if all vertices in its range
    # are in the same component. Otherwise 0. Leaves store the vertex index itself.
    id_val = [0] * (4 * N + 5) 

    def build_seg_tree(node_idx, lo, hi):
        if lo == hi:
            id_val[node_idx] = lo 
            return
        mid = (lo + hi) // 2
        build_seg_tree(2 * node_idx, lo, mid)
        build_seg_tree(2 * node_idx + 1, mid + 1, hi)
        # Initially, an internal node's children are distinct components, so id_val is 0
        id_val[node_idx] = 0
    
    if N > 0: # Build segment tree only if there are base vertices
        build_seg_tree(1, 1, N)

    total_mst_cost = 0
    # edges_count variable is not strictly needed if using num_components for final check

    def update_seg_tree(node_idx, lo, hi, query_L, query_R, U_vertex, cost_C):
        nonlocal total_mst_cost
        
        # If current segment [lo, hi] is outside query range [query_L, query_R]
        if lo > query_R or hi < query_L or lo > hi: # lo > hi for safety with N=0 or empty ranges
            return

        current_node_rep = id_val[node_idx]
        if current_node_rep != 0: 
            # This segment [lo, hi] is monochromatic (all its base vertices in one component).
            # Try to unite U_vertex with this component.
            if unite(U_vertex, current_node_rep):
                total_mst_cost += cost_C
            # Done with this segment for this query U_vertex, no need to go deeper.
            return
        
        # If current_node_rep is 0, this segment is mixed (or unoptimized).
        # If it's a leaf, it should have been non-zero (id_val[leaf] = vertex_idx).
        # So this block is primarily for internal nodes.
        if lo == hi: # Leaf node logic
            # This should be covered by the (current_node_rep != 0) case because leaves are non-zero.
            # If reached, it means id_val[leaf] was 0, which is unexpected.
            # For robustness: try to unite U_vertex with vertex 'lo'.
            if unite(U_vertex, lo): # 'lo' is the vertex index for this leaf
                total_mst_cost += cost_C
            return

        # Recursive step for internal node with id_val[node_idx] == 0
        mid = (lo + hi) // 2
        update_seg_tree(2 * node_idx, lo, mid, query_L, query_R, U_vertex, cost_C)
        update_seg_tree(2 * node_idx + 1, mid + 1, hi, query_L, query_R, U_vertex, cost_C)

        # After children are processed, update id_val[node_idx] for future queries.
        # If both children are monochromatic AND represent the same component,
        # then this node also becomes monochromatic.
        rep_left_child = id_val[2 * node_idx]
        rep_right_child = id_val[2 * node_idx + 1]

        if rep_left_child != 0 and rep_right_child != 0:
            if find(rep_left_child) == find(rep_right_child):
                id_val[node_idx] = rep_left_child # Store one of them as representative
                # Could also store find(rep_left_child) to always store a root,
                # but current DSU find handles non-root representatives fine.
        # else: id_val[node_idx] remains 0 or becomes 0 if it wasn't already.
        # If it was 0, it stays 0 if children aren't mergeable this way.

    # Handle N=0 case separately as segment tree assumes N >= 1.
    # Constraints: N, Q >= 1. So N cannot be 0.

    for query in queries_input:
        L, R, C, orig_idx = query['L'], query['R'], query['C'], query['orig_idx']
        # Map 0-indexed orig_idx to 1-indexed query vertex IDs N+1 to N+Q
        query_vertex_id = N + 1 + orig_idx 
        
        if N > 0: # Operations only make sense if there are base vertices
             update_seg_tree(1, 1, N, L, R, query_vertex_id, C)

    # An MST for (N+Q) vertices has (N+Q)-1 edges if connected.
    # Check connectivity using num_components.
    # num_components will be 1 if graph is connected.
    # If N=0 (not possible by constraints N>=1), then num_components would be Q.
    # If Q=0 (not possible by constraints Q>=1), then num_components would be N.
    if num_components == 1:
        print(total_mst_cost)
    else:
        print("-1")

solve()