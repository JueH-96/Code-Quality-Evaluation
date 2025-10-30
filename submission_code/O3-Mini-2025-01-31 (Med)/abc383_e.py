def main():
    import sys
    sys.setrecursionlimit(300000)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    k = int(next(it))
    
    # Read the graph edges.
    edges = []
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        w = int(next(it))
        edges.append((w, u, v))
    
    # Read the sequences A and B.
    A_list = [int(next(it)) for _ in range(k)]
    B_list = [int(next(it)) for _ in range(k)]
    # (It is guaranteed that no A_i equals any B_j.)
    
    # Observation:
    # For minimizing f(x, y) (the minimum possible "bottleneck" weight on
    # any path between x and y), one may show that the unique path between
    # x and y in a Minimum Spanning Tree (MST) attains f(x, y). Thus, we first
    # compute an MST of the given graph.
    
    # We use Kruskal's algorithm to compute the MST.
    parent_mst = list(range(n+1))
    rank_mst = [0]*(n+1)
    
    def find_mst(x):
        while parent_mst[x] != x:
            parent_mst[x] = parent_mst[parent_mst[x]]
            x = parent_mst[x]
        return x
    
    def union_mst(a, b):
        a = find_mst(a)
        b = find_mst(b)
        if a == b:
            return False
        if rank_mst[a] < rank_mst[b]:
            parent_mst[a] = b
        elif rank_mst[a] > rank_mst[b]:
            parent_mst[b] = a
        else:
            parent_mst[b] = a
            rank_mst[a] += 1
        return True

    # Sort edges by weight (increasing order).
    edges.sort(key=lambda x: x[0])
    mst_edges = []
    for w, u, v in edges:
        if union_mst(u, v):
            mst_edges.append((w, u, v))
            if len(mst_edges) == n - 1:
                break
                
    # Next, we address the pairing problem.
    #
    # We are given two sequences A of length K and B of length K. We can permute B
    # arbitrarily so that the sum of f(A_i, B_i) (with f(x, y) equal to the maximum edge on
    # the unique path between x and y in the MST) is minimized.
    #
    # A key observation is that in the MST, for any two vertices x and y,
    # f(x, y) is exactly the maximum edge weight along the unique path between them.
    #
    # Now suppose that we have a connected tree (the MST) and among its vertices, a multiset
    # A and a multiset B (note that the multisets A and B are disjoint by the problem condition).
    # We want to form a matching (i.e. pair each vertex from A with one from B) and our cost
    # for pairing vertices u and v is f(u, v). Since a vertex can appear only once in the matching,
    # one should “pair” vertices that become connected in the MST as early as possible (i.e. with
    # a small connecting edge weight). 
    #
    # How can we simulate this?
    # Consider processing the MST’s edges in increasing order. Initially, every vertex
    # is isolated. Then, when an edge of weight w connects two components, any pair that has one
    # vertex from one component and the other vertex from the other component now have a path
    # with maximum edge weight w. It is best to pair as many as possible of these vertices at
    # cost w, because pairing them later could only force us to use a higher cost edge.
    #
    # We use a DSU structure on the MST while also keeping track, for each component,
    # of:
    #   a_count: the number of vertices (or occurrences) from A in that component,
    #   b_count: the number of vertices from B in that component.
    #
    # When two components merge across an edge of weight w, we can pair:
    #   pairs = min(a_count_in_comp1, b_count_in_comp2) + min(b_count_in_comp1, a_count_in_comp2)
    #
    # Each such pairing will “use up” one vertex from A and one from B, and it will incur cost w.
    #
    # We then update the counts in the merged component by subtracting the matched ones.
    # (Every vertex can be used only once.) Since overall we have exactly K vertices in A and
    # K vertices in B, the matching will eventually cover all vertices.
    
    # Build DSU for matching computation on the MST.
    parent = list(range(n+1))
    # For each vertex, count how many times it appears in A (or B).
    a_count = [0]*(n+1)
    b_count = [0]*(n+1)
    
    for a in A_list:
        a_count[a] += 1
    for b in B_list:
        b_count[b] += 1
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y, w):
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return 0
        # Determine how many new pairs we can match when connecting these two components.
        # We can match:
        #   - min(a_count[rx], b_count[ry]) pairs (A from comp rx with B from comp ry)
        #   - min(b_count[rx], a_count[ry]) pairs (B from comp rx with A from comp ry)
        pairs = min(a_count[rx], b_count[ry]) + min(b_count[rx], a_count[ry])
        add_cost = pairs * w
        # After matching, these vertices are used up.
        # The remaining unmatched counts become:
        new_a = a_count[rx] + a_count[ry] - pairs
        new_b = b_count[rx] + b_count[ry] - pairs
        # Union the two components; for simplicity attach ry under rx.
        parent[ry] = rx
        a_count[rx] = new_a
        b_count[rx] = new_b
        return add_cost
    
    # Process the MST edges in increasing order of weight.
    # (Sort again in case the order from Kruskal is different.)
    mst_edges.sort(key=lambda x: x[0])
    total_cost = 0
    for w, u, v in mst_edges:
        total_cost += union(u, v, w)
    
    sys.stdout.write(str(total_cost))
    
if __name__ == '__main__':
    main()