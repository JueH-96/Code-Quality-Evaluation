from collections import deque

def solve():
    n, x = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    p = list(map(int, input().split()))
    q = list(map(int, input().split()))

    x -= 1
    
    if all(a[i] == 0 and b[i] == 0 for i in range(n) if i != x):
        print(0)
        return

    graph = [[] for _ in range(n)]
    for i in range(n):
        if i != x:
            graph[p[i]-1].append(i)
            graph[q[i]-1].append(i)

    q = deque()
    visited = [False] * n
    dist = [-1] * n

    for i in range(n):
        if i != x and (a[i] > 0 or b[i] > 0):
            q.append(i)
            visited[i] = True
            dist[i] = 1

    while q:
        u = q.popleft()
        for v in range(n):
            if v != x and (p[v]-1 == u or q[v]-1 == u) and not visited[v]:
                q.append(v)
                visited[v] = True
                dist[v] = dist[u] + 1

    ans = -1
    if all(a[i] == 0 and b[i] == 0 for i in range(n) if i != x):
        max_dist = 0
        for i in range(n):
            if i != x and dist[i] > 0:
                max_dist = max(max_dist, dist[i])
        ans = max_dist
    else:
        reachable = [False] * n
        q = deque()
        for i in range(n):
            if i != x and (a[i] > 0 or b[i] > 0):
                q.append(i)
                reachable[i] = True

        while q:
            u = q.popleft()
            for v in range(n):
                if v != x and (p[v]-1 == u or q[v]-1 == u) and not reachable[v]:
                    q.append(v)
                    reachable[v] = True
        
        possible = True
        for i in range(n):
            if i != x and (a[i] > 0 or b[i] > 0) and not reachable[i]:
                possible = False
                break
        
        if not possible:
            ans = -1
        else:
            
            q = deque()
            visited = [False] * n
            dist = [-1] * n

            for i in range(n):
                if i != x and (a[i] > 0 or b[i] > 0):
                    q.append(i)
                    visited[i] = True
                    dist[i] = 1

            while q:
                u = q.popleft()
                for v in range(n):
                    if v != x and not visited[v]:
                        if p[v]-1 == u or q[v]-1 == u:
                            q.append(v)
                            visited[v] = True
                            dist[v] = dist[u] + 1
            
            if all(a[i] == 0 and b[i] == 0 for i in range(n) if i != x):
                max_dist = 0
                for i in range(n):
                    if i != x and dist[i] > 0:
                        max_dist = max(max_dist, dist[i])
                ans = max_dist
            else:
                ans = -1

    print(ans)

solve()