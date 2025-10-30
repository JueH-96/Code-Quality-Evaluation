# YOUR CODE HERE
import sys

def solve():
    n = int(sys.stdin.readline())
    adj = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v = map(int, sys.stdin.readline().split())
        u -= 1
        v -= 1
        adj[u].append(v)
        adj[v].append(u)

    leaves = []
    for i in range(n):
        if len(adj[i]) == 1:
            leaves.append(i)

    stars = []
    visited = [False] * n

    def dfs(u, count):
        visited[u] = True
        count += 1
        for v in adj[u]:
            if not visited[v]:
                count = dfs(v, count)
        return count
    
    while leaves:
        new_leaves = []
        for leaf in leaves:
            if not visited[leaf]:
                star_size = dfs(leaf, 0)
                stars.append(star_size -1)
                
        for i in range(n):
            if not visited[i]:
                count = 0
                for neighbor in adj[i]:
                    if visited[neighbor]:
                        count += 1
                if count == len(adj[i]) -1 and len(adj[i]) > 1:
                    new_leaves.append(i)
                    
        leaves = new_leaves
        

    stars.sort()
    print(*stars)

solve()