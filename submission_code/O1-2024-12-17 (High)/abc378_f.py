def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    
    # Read number of vertices
    N = int(input())
    
    # Adjacency list and degree array
    adj = [[] for _ in range(N)]
    deg = [0]*N
    
    edges = []
    for _ in range(N-1):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        edges.append((u, v))
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1
    
    # Collect vertices whose degree is exactly 2 or 3
    s2_nodes = [i for i in range(N) if deg[i] == 2]
    s3_nodes = [i for i in range(N) if deg[i] == 3]
    
    # If there are no s3_nodes, we cannot form any cycle where all interior nodes have degree 3
    # so the answer would be 0. But let's proceed with the general solution, which will
    # also give 0 in that case.
    
    # We will build a "union-find/DSU" over the S3 subgraph to identify connected components
    # of pure degree-3 vertices.
    
    # Map each s3-node to a consecutive index [0..len(s3_nodes)-1]
    s3_index = [-1]*N
    for i, node in enumerate(s3_nodes):
        s3_index[node] = i
    
    size_s3 = len(s3_nodes)
    
    # Union-Find (Disjoint Set) for the S3 subgraph
    parent = list(range(size_s3))
    rank_ = [0]*size_s3
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if rank_[ra] < rank_[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank_[ra] == rank_[rb]:
                rank_[ra] += 1
    
    # For each edge that connects two s3-nodes, unite them
    for (u, v) in edges:
        if s3_index[u] != -1 and s3_index[v] != -1:
            union(s3_index[u], s3_index[v])
    
    # Compress and label each s3-node by its connected component
    for i in range(size_s3):
        find(i)
    
    # Map each root to a new integer component label
    comp_map = {}
    comp_count = 0
    
    # comp_of_s3[v] will hold the connected component index of the s3-node v
    comp_of_s3 = [-1]*N
    for i, node in enumerate(s3_nodes):
        root = find(i)
        if root not in comp_map:
            comp_map[root] = comp_count
            comp_count += 1
        comp_of_s3[node] = comp_map[root]
    
    # Now, each s3-node v belongs to some comp_of_s3[v] in [0..comp_count-1].
    # Next, for each s2-node x, we gather which s3-components it can directly "enter"
    # by looking at x's neighbors that are s3-nodes.  This yields a small set (up to 2)
    # of component indices.  Call this compSet(x).
    
    from collections import defaultdict
    
    # For convenience, we'll store for each component c, the list of s2-nodes that can connect to c.
    Lc = [[] for _ in range(comp_count)]
    # Also, if an s2-node has exactly two distinct s3-components {c1, c2}, we will store it in
    # a dictionary keyed by (min(c1,c2), max(c1,c2)), to track potential double-counting.
    double_map = defaultdict(list)
    
    # We'll keep comp_set_of[x] = None if x cannot connect to any s3-component,
    # otherwise a tuple (c,) or (c1,c2).
    comp_set_of = [None]*N
    
    for x in s2_nodes:
        neighbor_comps = set()
        for nei in adj[x]:
            if deg[nei] == 3:
                neighbor_comps.add(comp_of_s3[nei])
        
        # Convert that set to a sorted tuple
        if len(neighbor_comps) == 0:
            # x has no s3 neighbor => cannot form a valid cycle interior
            comp_set_of[x] = None
        elif len(neighbor_comps) == 1:
            c = next(iter(neighbor_comps))
            comp_set_of[x] = (c,)
            Lc[c].append(x)
        else:
            # up to 2 distinct components for deg=2-based neighbors
            sorted_comps = sorted(neighbor_comps)
            c1, c2 = sorted_comps[0], sorted_comps[1]
            comp_set_of[x] = (c1, c2)
            Lc[c1].append(x)
            Lc[c2].append(x)
            double_map[(c1, c2)].append(x)
    
    def comb2(a):
        return a*(a-1)//2 if a >= 2 else 0
    
    # Count all pairs by summing combination of each Lc
    single_count = 0
    for c in range(comp_count):
        single_count += comb2(len(Lc[c]))
    
    # Subtract out the double counts (if a node is in exactly two components c1,c2,
    # that pair of nodes is counted in Lc1 and in Lc2, so we fix by subtracting once)
    double_count = 0
    for (c1c2, arr) in double_map.items():
        double_count += comb2(len(arr))
    
    total = single_count - double_count
    
    # Now, we must exclude pairs of s2-nodes that are neighbors in the original tree:
    # adding an edge between them would create a multi-edge (not simple).
    # If two s2 nodes are neighbors, we check if they "share" a component in comp_set_of[].
    # If they do, that pair was counted, so we subtract one for each such edge.
    
    disallowed = 0
    for (u, v) in edges:
        if deg[u] == 2 and deg[v] == 2:
            cu = comp_set_of[u]
            cv = comp_set_of[v]
            if cu and cv:
                # Check if intersection is non-empty
                # cu, cv can each have size=1 or 2
                # We do a small intersection check in O(1) time
                if len(cu) == 1 and len(cv) == 1:
                    if cu[0] == cv[0]:
                        disallowed += 1
                elif len(cu) == 1 and len(cv) == 2:
                    if cu[0] in cv:
                        disallowed += 1
                elif len(cu) == 2 and len(cv) == 1:
                    if cv[0] in cu:
                        disallowed += 1
                else:  # both length 2
                    # Intersection is non-empty if any component is common
                    if cu[0] in cv or cu[1] in cv:
                        disallowed += 1
    
    total -= disallowed
    
    print(total)

# Do not forget to call main()!
if __name__ == "__main__":
    main()