def main():
    import sys
    from collections import deque
    
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n1 = int(next(it))
    n2 = int(next(it))
    m  = int(next(it))
    
    # Build separate graphs for the two connected components.
    # For component 1, vertices are 1..n1. We will represent them as indices 0..n1-1.
    # For component 2, vertices are n1+1..n1+n2. We represent them as indices 0..n2-1.
    graph1 = [[] for _ in range(n1)]
    graph2 = [[] for _ in range(n2)]
    
    for _ in range(m):
        a = int(next(it))
        b = int(next(it))
        # Use the fact that there is no edge connecting the two parts.
        if a <= n1 and b <= n1:
            # Both vertices are in the first component.
            graph1[a-1].append(b-1)
            graph1[b-1].append(a-1)
        elif a > n1 and b > n1:
            # Both vertices are in the second component.
            # Adjust indices: subtract n1 and use 0-indexing.
            u = a - n1 - 1
            v = b - n1 - 1
            graph2[u].append(v)
            graph2[v].append(u)
    
    # In comp1, we compute distances from vertex 1 (index 0).
    dist1 = [-1] * n1
    dq = deque()
    dq.append(0)
    dist1[0] = 0
    while dq:
        cur = dq.popleft()
        for nxt in graph1[cur]:
            if dist1[nxt] == -1:
                dist1[nxt] = dist1[cur] + 1
                dq.append(nxt)
    max_d1 = max(dist1)
    
    # In comp2, we compute distances from vertex (n1+n2), which is the last vertex in comp2.
    # On our adjusted indexing for comp2, that is vertex (n2-1).
    dist2 = [-1] * n2
    dq = deque()
    start = n2 - 1
    dq.append(start)
    dist2[start] = 0
    while dq:
        cur = dq.popleft()
        for nxt in graph2[cur]:
            if dist2[nxt] == -1:
                dist2[nxt] = dist2[cur] + 1
                dq.append(nxt)
    max_d2 = max(dist2)
    
    # By adding one edge connecting some vertex u (in component 1) and some vertex v (in component 2),
    # the shortest path between vertex 1 and vertex (n1+n2) becomes:
    #   d = (distance from 1 to u) + 1 (new edge) + (distance from v to n1+n2)
    # To maximize d, we choose u and v that maximize these distances.
    # So the maximum d is: max_d1 + 1 + max_d2.
    ans = max_d1 + max_d2 + 1
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()