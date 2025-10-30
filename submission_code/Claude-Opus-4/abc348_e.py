from collections import defaultdict, deque

def solve():
    n = int(input())
    
    if n == 1:
        print(0)
        return
    
    # Build adjacency list
    adj = defaultdict(list)
    for _ in range(n - 1):
        a, b = map(int, input().split())
        adj[a].append(b)
        adj[b].append(a)
    
    c = list(map(int, input().split()))
    
    # Calculate f(1) using BFS
    dist = [-1] * (n + 1)
    dist[1] = 0
    queue = deque([1])
    
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    
    f1 = sum(c[i-1] * dist[i] for i in range(1, n + 1))
    
    # Calculate subtree sums for each vertex
    subtree_sum = [0] * (n + 1)
    visited = [False] * (n + 1)
    
    def dfs_subtree(u, parent):
        visited[u] = True
        subtree_sum[u] = c[u-1]
        for v in adj[u]:
            if v != parent and not visited[v]:
                dfs_subtree(v, u)
                subtree_sum[u] += subtree_sum[v]
    
    dfs_subtree(1, -1)
    
    # Calculate f(v) for all vertices using DFS
    f_values = [0] * (n + 1)
    f_values[1] = f1
    visited = [False] * (n + 1)
    
    def dfs_calculate(u, parent):
        visited[u] = True
        for v in adj[u]:
            if v != parent and not visited[v]:
                # When moving from u to v:
                # Vertices in subtree of v get closer by 1
                # Other vertices get farther by 1
                total_sum = sum(c)
                f_values[v] = f_values[u] + total_sum - 2 * subtree_sum[v]
                dfs_calculate(v, u)
    
    dfs_calculate(1, -1)
    
    # Find minimum f value
    min_f = min(f_values[1:n+1])
    print(min_f)

solve()