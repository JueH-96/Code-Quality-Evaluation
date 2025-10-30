def solve():
    n, m = map(int, input().split())
    sets = []
    for _ in range(n):
        a = list(map(int, input().split()))
        s = set(map(int, input().split()))
        sets.append(s)

    if any(1 in s and m in s for s in sets):
        print(0)
        return

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if any(x in sets[j] for x in sets[i]):
                adj[i].append(j)
                adj[j].append(i)

    q = []
    dist = [-1] * n
    
    start_nodes = []
    end_nodes = []

    for i in range(n):
        if 1 in sets[i]:
            start_nodes.append(i)
        if m in sets[i]:
            end_nodes.append(i)

    for start_node in start_nodes:
        q.append(start_node)
        dist[start_node] = 0
    
    
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)

    ans = float('inf')
    for end_node in end_nodes:
      if dist[end_node] != -1:
        ans = min(ans, dist[end_node])

    if ans == float('inf'):
        print(-1)
    else:
        print(ans)

solve()