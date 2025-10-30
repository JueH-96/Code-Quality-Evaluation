import sys

# Set recursion limit for deep segment tree calls
sys.setrecursionlimit(2 * 10**5 + 500) # N can be up to 2e5, segment tree depth is log N

# DSU (Disjoint Set Union) class
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        # self.num_components is not directly used for the final connectivity check of N+Q vertices
        # but is useful for general DSU tracking if needed.

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Union by size heuristic
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            return True
        return False

def solve():
    N, Q = map(int, sys.stdin.readline().split())

    queries = []
    for _ in range(Q):
        L, R, C = map(int, sys.stdin.readline().split())
        queries.append((L, R, C))

    # Calculate max possible DSU nodes: N (primary) + Q (query) + ~4*N (segment tree nodes)
    # 4*N is a safe upper bound for segment tree nodes when N is not a power of 2.
    MAX_DSU_NODES = N + Q + 4 * N + 5 # Add a small buffer
    dsu = DSU(MAX_DSU_NODES)
    all_edges = []

    # DSU ID mapping:
    # Primary vertices 1..N map to DSU IDs 0..N-1
    # Query vertices N+1..N+Q map to DSU IDs N..N+Q-1
    # Segment tree nodes will use DSU IDs starting from N+Q

    seg_node_dsu_id_counter = N + Q # Next available DSU ID for a segment tree node

    # This array stores the DSU ID assigned to each segment tree node,
    # indexed by its `tree_node_idx` (1 for root, 2*idx for left child, etc.)
    # Max `tree_node_idx` is approx 4*N for N leaves.
    seg_tree_node_dsu_ids = [-1] * (4 * N + 5) 

    # Recursive function to build the segment tree and add 0-cost edges
    # `node_idx_in_tree` is the conceptual index in a complete binary tree (1-based)
    # `current_L`, `current_R` define the range covered by this segment tree node
    def build_seg_tree(node_idx_in_tree, current_L, current_R):
        nonlocal seg_node_dsu_id_counter
        
        # Assign a unique DSU ID to this segment tree node
        dsu_id_for_this_seg_node = seg_node_dsu_id_counter
        seg_tree_node_dsu_ids[node_idx_in_tree] = dsu_id_for_this_seg_node
        seg_node_dsu_id_counter += 1

        if current_L == current_R: # This is a leaf node in the segment tree
            # Connect this segment tree leaf node to its corresponding actual graph vertex
            # Vertex `current_L` (1-indexed) maps to DSU ID `current_L - 1`
            all_edges.append((dsu_id_for_this_seg_node, current_L - 1, 0))
        else: # This is an internal node
            mid = (current_L + current_R) // 2
            left_child_dsu_id = build_seg_tree(2 * node_idx_in_tree, current_L, mid)
            right_child_dsu_id = build_seg_tree(2 * node_idx_in_tree + 1, mid + 1, current_R)
            
            # Connect the parent segment tree node to its children with 0 cost
            # This ensures that if you connect to the parent node, you also connect to all
            # original vertices covered by its children, effectively grouping them.
            all_edges.append((dsu_id_for_this_seg_node, left_child_dsu_id, 0))
            all_edges.append((dsu_id_for_this_seg_node, right_child_dsu_id, 0))
        
        return dsu_id_for_this_seg_node

    # Build the segment tree for the range [1, N]
    # This step is crucial for efficient range connections.
    if N > 0: # Ensure N is positive as per constraints (N>=1)
        build_seg_tree(1, 1, N)

    # Process each query and add corresponding edges
    for q_idx, (L_query, R_query, cost) in enumerate(queries):
        # The query vertex N+i (1-indexed) maps to DSU ID N + (q_idx+1) - 1 = N + q_idx
        query_vertex_dsu_id = N + q_idx 

        # Recursive function to traverse the segment tree and add edges for the current query
        def add_query_edges(node_idx_in_tree, current_L, current_R):
            # Get the DSU ID that this segment tree node corresponds to
            dsu_node_id = seg_tree_node_dsu_ids[node_idx_in_tree]

            # Case 1: Current segment tree node's range is fully contained within the query range [L_query, R_query]
            if L_query <= current_L and current_R <= R_query:
                # Add an edge between the query vertex and this segment tree node.
                # Since this segment tree node effectively represents all vertices in its range (due to 0-cost edges),
                # this single edge connects the query vertex to the entire covered range.
                all_edges.append((query_vertex_dsu_id, dsu_node_id, cost))
            # Case 2: Current segment tree node's range is completely outside the query range
            elif current_R < L_query or current_L > R_query:
                pass # No overlap, do nothing
            # Case 3: Current segment tree node's range partially overlaps the query range
            else:
                mid = (current_L + current_R) // 2
                add_query_edges(2 * node_idx_in_tree, current_L, mid) # Recurse on left child
                add_query_edges(2 * node_idx_in_tree + 1, mid + 1, current_R) # Recurse on right child

        if N > 0: # Only add query edges if N > 0 (i.e., there are primary vertices to connect to)
            add_query_edges(1, 1, N)

    # Sort all collected edges by cost in ascending order
    # This is a prerequisite for Kruskal's algorithm.
    all_edges.sort(key=lambda x: x[2]) # Edge format: (u, v, cost)

    mst_cost = 0

    # Run Kruskal's algorithm
    for u, v, cost in all_edges:
        # If adding this edge connects two previously disconnected components, add it to MST
        if dsu.union(u, v):
            mst_cost += cost

    # Connectivity Check:
    # The graph is connected if all N+Q original vertices (DSU IDs 0 to N+Q-1)
    # belong to the same connected component.
    
    if N + Q == 0: # Edge case: no vertices in graph (not possible by constraints N, Q >= 1)
        print(0)
        return

    distinct_roots = set()
    for i in range(N + Q): # Iterate through DSU IDs for the N primary and Q query vertices
        distinct_roots.add(dsu.find(i))

    # If there is only one distinct root among these N+Q vertices, the graph is connected.
    if len(distinct_roots) == 1:
        print(mst_cost)
    else:
        print(-1)

# Standard input/output handling
if __name__ == '__main__':
    solve()