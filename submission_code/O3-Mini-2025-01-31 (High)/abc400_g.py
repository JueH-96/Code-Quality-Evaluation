# YOUR CODE HERE
def main():
    import sys
    # We use the networkx library to perform maximum‐weight matching.
    # (This solution is based on the observation that one may restrict attention
    # to “candidate” edges obtained from adjacent nodes in three different orders.)
    try:
        import networkx as nx
    except ImportError:
        sys.stderr.write("This solution requires networkx.
")
        return

    data = sys.stdin.buffer.read().split()
    if not data:
        return
    t = int(data[0])
    pos = 1
    out_lines = []
    for _ in range(t):
        if pos >= len(data): break
        N = int(data[pos]); pos+=1
        K = int(data[pos]); pos+=1
        X = [0]*N
        Y = [0]*N
        Z = [0]*N
        for i in range(N):
            X[i] = int(data[pos]); Y[i] = int(data[pos+1]); Z[i] = int(data[pos+2])
            pos += 3

        # Build candidate edges from adjacent neighbors in each sorted order.
        # For each ordering (X, Y, Z) we sort by that coordinate descending.
        # For consecutive indices (in that order) add an edge between the two cakes.
        # The “price” on an edge (u,v) is defined as
        #      max( X[u]+X[v], Y[u]+Y[v], Z[u]+Z[v] ).
        cand_edges = {}  # key: (u,v) with u < v, value = best price for that pair.
        for arr in (X, Y, Z):
            order = sorted(range(N), key=lambda i: arr[i], reverse=True)
            for i in range(len(order)-1):
                u = order[i]
                v = order[i+1]
                if u > v:
                    u, v = v, u
                w = X[u] + X[v]
                temp = Y[u] + Y[v]
                if temp > w:
                    w = temp
                temp = Z[u] + Z[v]
                if temp > w:
                    w = temp
                key = (u, v)
                if key in cand_edges:
                    if w > cand_edges[key]:
                        cand_edges[key] = w
                else:
                    cand_edges[key] = w

        # Build a graph from 0 .. N-1 and add these edges.
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for (u, v), w in cand_edges.items():
            G.add_edge(u, v, weight=w)

        # We compute two matchings:
        #  1) The usual maximum–weight matching (which might use fewer than N/2 pairs)
        #  2) The maximum–cardinality matching (with maximum weight among those)
        # One of these matchings will “contain” the optimal k–matching.
        match1 = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False)
        match2 = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
        
        def matching_to_weights(mset):
            # Each edge in mset is a frozenset of two vertices.
            # We convert it to (u,v) with u<v and return its corresponding weight.
            res = []
            for edge in mset:
                u, v = sorted(edge)
                w = cand_edges.get((u, v), G[u][v]['weight'])
                res.append(w)
            return res

        cand_sum = -1  # will hold the best total sum for K pairs.
        lis1 = matching_to_weights(match1)
        lis2 = matching_to_weights(match2)
        if len(lis1) >= K:
            lis1.sort(reverse=True)
            s = sum(lis1[:K])
            if s > cand_sum:
                cand_sum = s
        if len(lis2) >= K:
            lis2.sort(reverse=True)
            s = sum(lis2[:K])
            if s > cand_sum:
                cand_sum = s
        # Fallback (should almost never be needed):
        if cand_sum < 0:
            used = [False]*N
            selected = []
            sorted_edges = sorted(cand_edges.items(), key=lambda kv: kv[1], reverse=True)
            for (u,v), w in sorted_edges:
                if not used[u] and not used[v]:
                    used[u] = used[v] = True
                    selected.append(w)
            selected.sort(reverse=True)
            if len(selected) < K:
                cand_sum = sum(selected)
            else:
                cand_sum = sum(selected[:K])
        out_lines.append(str(cand_sum))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()