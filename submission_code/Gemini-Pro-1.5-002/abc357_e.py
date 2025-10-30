# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    adj = [[] for _ in range(n)]
    for i in range(n):
        adj[i].append(a[i] - 1)
    
    count = 0
    for start_node in range(n):
        reachable = set()
        q = [start_node]
        visited = [False] * n
        
        while q:
            u = q.pop(0)
            if visited[u]:
                continue
            visited[u] = True
            reachable.add(u)
            for v in adj[u]:
                if not visited[v]:
                    q.append(v)
        count += len(reachable)
    print(count)

solve()