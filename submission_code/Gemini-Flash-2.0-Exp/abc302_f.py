def solve():
    n, m = map(int, input().split())
    sets = []
    for _ in range(n):
        a = int(input())
        s = list(map(int, input().split()))
        sets.append(set(s))

    if any(1 in s and m in s for s in sets):
        print(0)
        return

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if len(sets[i].intersection(sets[j])) > 0:
                adj[i].append(j)
                adj[j].append(i)

    q = []
    dist = [-1] * n
    
    for i in range(n):
        if 1 in sets[i]:
            q.append(i)
            dist[i] = 1

    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)

    ans = float('inf')
    for i in range(n):
        if m in sets[i] and dist[i] != -1:
            ans = min(ans, dist[i])

    if ans == float('inf'):
        print(-1)
    else:
        print(ans - 1)

solve()