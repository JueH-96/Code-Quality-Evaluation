def main():
    import sys, collections
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n1 = int(next(it))
    n2 = int(next(it))
    m  = int(next(it))
    
    # The graph consists of exactly two connected components:
    # Group 1: vertices 1 .. n1 (they are connected)
    # Group 2: vertices n1+1 .. n1+n2 (they are connected)
    # It is given that no edge connects the two groups.
    #
    # When we add exactly one edge connecting a vertex u in group 1 and a vertex v in group 2,
    # the unique way from vertex 1 to vertex (n1+n2) must go:
    #   Path from 1 to u (in group1) + the new edge (u,v) + path from v to (n1+n2) (in group2).
    # Thus, the shortest path distance in the augmented graph is:
    #   d(1,u) + 1 + d(v, n1+n2)
    # To maximize this shortest distance we can choose u and v independently.
    # In group1, we choose u so that the distance from 1, d1(u), is as large as possible.
    # In group2, we choose v so that the distance from (n1+n2), d2(v), is as large as possible.
    #
    # Hence, if we let D1 be the maximum distance from 1 in group1
    # and D2 be the maximum distance from (n1+n2) in group2,
    # the maximum possible d is: D1 + 1 + D2.
    #
    # Our task is to compute D1 and D2.
    
    # Build separate graphs for group1 and group2.
    # For group1, use 0-indexing: vertices 0 to n1-1 correspond to 1..n1.
    g1 = [[] for _ in range(n1)]
    # For group2, map vertices n1+1..n1+n2 to indices 0..n2-1.
    g2 = [[] for _ in range(n2)]
    
    for _ in range(m):
        a = int(next(it))
        b = int(next(it))
        if a <= n1 and b <= n1:
            # Both vertices are in group1.
            u = a - 1
            v = b - 1
            g1[u].append(v)
            g1[v].append(u)
        elif a > n1 and b > n1:
            # Both vertices are in group2.
            u = a - n1 - 1  # map: a -> index in 0...n2-1
            v = b - n1 - 1
            g2[u].append(v)
            g2[v].append(u)
        else:
            # If an edge were to connect the groups, then vertex 1 and vertex (n1+n2)
            # would be connected in the original graph. The problem guarantee forbids this.
            pass
    
    # BFS in group1 from vertex 1 (index 0) to compute distances.
    dist1 = [-1] * n1
    dq = collections.deque()
    dist1[0] = 0
    dq.append(0)
    while dq:
        cur = dq.popleft()
        for nxt in g1[cur]:
            if dist1[nxt] == -1:
                dist1[nxt] = dist1[cur] + 1
                dq.append(nxt)
    max_d1 = max(dist1)
    
    # BFS in group2 from vertex (n1+n2) which corresponds to index n2-1 in g2.
    dist2 = [-1] * n2
    dq = collections.deque()
    start = n2 - 1  # mapping (n1+n2) -> index n2-1
    dist2[start] = 0
    dq.append(start)
    while dq:
        cur = dq.popleft()
        for nxt in g2[cur]:
            if dist2[nxt] == -1:
                dist2[nxt] = dist2[cur] + 1
                dq.append(nxt)
    max_d2 = max(dist2)
    
    # The maximum shortest path length after adding one edge is:
    #   max_d1 + max_d2 + 1
    ans = max_d1 + max_d2 + 1
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()