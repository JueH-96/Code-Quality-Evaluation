# YOUR CODE HERE
import sys

# Increase recursion depth for deep segment tree operations
# Setting recursion depth is important for deep recursive calls, typical in segment trees
# for large N. Python's default limit might be too low.
try:
    # Set recursion depth large enough for N up to 2*10^5
    # A value like N + buffer should be safe for most recursive functions like DSU find or segment tree ops.
     sys.setrecursionlimit(2 * 10**5 + 100) 
except Exception as e: 
     # print(f"Could not set recursion depth: {e}", file=sys.stderr) # Debug message optional
     pass # Continue with default limits if setting fails


# Fast I/O is recommended for large inputs, especially competitive programming
# Using sys.stdin.readline is generally faster than input()

def solve():
    N, Q = map(int, sys.stdin.readline().split())

    operations = []
    for i in range(Q):
        L, R, C = map(int, sys.stdin.readline().split())
        # Store 1-based vertex ID for auxiliary node N+1+i
        operations.append({'L': L, 'R': R, 'C': C, 'id': N + 1 + i})

    # Compute costs for potential edges (j, j+1) using a segment tree.
    # The cost of edge (j, j+1) is the minimum cost C_k of any operation k
    # such that the operation's range [L_k, R_k] covers both j and j+1.
    # This means L_k <= j and j+1 <= R_k, or equivalently L_k <= j < R_k.
    
    INF = float('inf')
    
    # The segment tree will cover indices 0 to N-2, representing the N-1 boundaries
    # between vertices j and j+1 for j from 1 to N-1.
    # Boundary j <-> j+1 corresponds to index j-1.
    seg_N = N - 1 # Number of boundaries
    adj_costs_list = [] # To store computed costs for (j, j+1) edges

    if seg_N > 0: # Only build segment tree if N > 1 (i.e., there is at least one boundary)
        # Calculate minimum power of 2 size for segment tree array base
        seg_tree_base_size = 1
        while seg_tree_base_size < seg_N:
            seg_tree_base_size *= 2
        
        # Initialize segment tree arrays
        # seg_tree stores the minimum cost found for the range
        seg_tree = [INF] * (2 * seg_tree_base_size)
        # lazy array stores pending minimum updates to be propagated down
        lazy = [INF] * (2 * seg_tree_base_size) 

        # Function to push lazy updates down from a node to its children
        def push(node):
            if lazy[node] == INF: return # No pending update at this node
            
            # Calculate indices of children nodes
            child1_idx = node * 2
            child2_idx = node * 2 + 1
            
            # Apply lazy value to children if they exist within array bounds
            if child1_idx < len(seg_tree):
                seg_tree[child1_idx] = min(seg_tree[child1_idx], lazy[node])
                lazy[child1_idx] = min(lazy[child1_idx], lazy[node]) # Propagate minimum update
            if child2_idx < len(seg_tree):
                 seg_tree[child2_idx] = min(seg_tree[child2_idx], lazy[node])
                 lazy[child2_idx] = min(lazy[child2_idx], lazy[node]) # Propagate minimum update
            
            lazy[node] = INF # Reset lazy value for the current node after pushing

        # Function for performing range minimum update on the segment tree
        # Updates range [L, R] (inclusive) with value `val`. Node `node` covers range [start, end].
        def range_update(node, start, end, L, R, val):
            if R < start or end < L: # Query range is completely outside node's range
                return
            
            if L <= start and end <= R: # Node's range is fully within query range
                # Apply update to this node and mark it lazy
                seg_tree[node] = min(seg_tree[node], val)
                lazy[node] = min(lazy[node], val)
                return

            # Node's range partially overlaps query range or contains it
            push(node) # Ensure lazy updates are pushed down before recursing
            
            mid = (start + end) // 2
            # Recurse on children
            range_update(node*2, start, mid, L, R, val)
            range_update(node*2+1, mid + 1, end, L, R, val)
            
            # Update current node value based on children values (optional, mainly for range queries)
            # Not strictly needed if we only query leaf values eventually.
            # seg_tree[node] = min(seg_tree[node*2], seg_tree[node*2+1])


        # Process all operations to populate the segment tree with minimum costs
        for op in operations:
            L, R, C = op['L'], op['R'], op['C']
            # The operation (L, R, C) potentially sets the minimum cost for boundaries j <-> j+1 
            # where L <= j < R. This corresponds to 0-based indices [L-1, R-2].
            if L < R: # Ensure the range covers at least one boundary
                 range_update(1, 0, seg_tree_base_size - 1, L - 1, R - 2, C)

        # Query all leaf nodes to retrieve the final minimum costs for each boundary
        # Traverses the tree recursively down to the leaves.
        def query_leaves(node, start, end):
             if start == end: # Reached a leaf node
                 # Check if this leaf corresponds to a valid boundary index (0 to N-2)
                 if start < seg_N: 
                     # Boundary j <-> j+1 corresponds to leaf index `start = j-1`.
                     # The vertices connected by this boundary are j = start+1 and j+1 = start+2.
                     adj_costs_list.append({'u': start + 1, 'v': start + 2, 'cost': seg_tree[node]})
                 return

             # This is an internal node, push down any pending lazy updates
             push(node) 
             mid = (start + end) // 2
             # Recurse on children to reach leaves
             query_leaves(node*2, start, mid)
             query_leaves(node*2+1, mid + 1, end)
        
        # Start querying leaves process from the root node (node 1)
        query_leaves(1, 0, seg_tree_base_size - 1)

    # Collect all potential edges for Kruskal's algorithm
    edges = []
    # Type 1 edges: (N+i, L_i, C_i) - connecting auxiliary node N+i to the first vertex L_i in its range.
    # This edge represents the connection potential of the auxiliary node.
    for op in operations:
        edges.append({'u': op['id'], 'v': op['L'], 'cost': op['C']})
    
    # Type 2 edges: (j, j+1, C'_{j, j+1}) - edges between adjacent vertices 1..N,
    # with cost derived from the minimum cost operation covering that boundary.
    for edge_data in adj_costs_list:
         if edge_data['cost'] != INF: # Only include edges if a finite cost was assigned (i.e., boundary covered by at least one operation)
             edges.append({'u': edge_data['u'], 'v': edge_data['v'], 'cost': edge_data['cost']})

    # Sort all collected edges by cost non-decreasingly for Kruskal's algorithm
    edges.sort(key=lambda x: x['cost'])

    # Disjoint Set Union (DSU) data structure setup
    total_vertices = N + Q
    # Use 1-based indexing for vertices 1..N+Q. Array size needs to be N+Q+1.
    parent = list(range(total_vertices + 1)) 
    size = [1] * (total_vertices + 1) # Used for union by size heuristic
    
    # Find operation with path compression optimization
    def find(i):
        # Find the root of the set containing element i
        root = i
        while parent[root] != root:
           root = parent[root]
        # Path compression: make all nodes on the path point directly to the root
        curr = i
        while parent[curr] != root:
           next_node = parent[curr]
           parent[curr] = root
           curr = next_node
        return root

    # Union operation with union by size heuristic
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            # Merge the smaller set into the larger set
            if size[root_i] < size[root_j]:
                root_i, root_j = root_j, root_i # Ensure root_i points to the root of the larger set
            parent[root_j] = root_i
            size[root_i] += size[root_j]
            return True # Return True indicating a successful merge occurred
        return False # Return False if i and j were already in the same set

    # Kruskal's algorithm execution
    mst_cost = 0
    edges_count = 0
    
    # Iterate through sorted edges and add to the Minimum Spanning Tree (MST) if they connect distinct components
    for edge in edges:
        # If the endpoints u and v of the edge are in different sets (components)
        if union(edge['u'], edge['v']):
            mst_cost += edge['cost'] # Add edge cost to total MST cost
            edges_count += 1 # Increment count of edges added to MST
            
            # Optimization: If we have added V-1 edges, the MST is complete (assuming the graph is connected)
            # Check requires total_vertices > 0 to avoid division by zero issues or index errors in edge cases.
            if total_vertices > 0 and edges_count == total_vertices - 1:
                 break

    # Final check for connectivity and output result
    if total_vertices == 0: # Handle edge case N=0, Q=0 (though constraints say N >= 1)
         print(0)
    elif total_vertices == 1: # Handle edge case N=1, Q=0 (single vertex graph)
         print(0)
    # A graph with V vertices is connected if and only if its spanning forest contains exactly V-1 edges.
    # In that case, the spanning forest is a single spanning tree.
    elif edges_count == total_vertices - 1: 
         print(mst_cost) # Graph is connected, print the calculated MST cost
    else: # If edges_count < V-1, the graph is disconnected
         print("-1")

solve()