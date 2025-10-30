from collections import deque

def solve():
    n, m = map(int, input().split())
    sets = []
    for _ in range(n):
        input()
        sets.append(set(map(int, input().split())))

    if any(1 in s and m in s for s in sets):
        print(0)
        return

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if sets[i].intersection(sets[j]):
                adj[i].append(j)
                adj[j].append(i)

    q = deque()
    dist = [-1] * n
    for i in range(n):
        if 1 in sets[i]:
            q.append(i)
            dist[i] = 0

    while q:
        u = q.popleft()
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
        print(ans)

solve()