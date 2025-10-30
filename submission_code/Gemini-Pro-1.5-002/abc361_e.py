# YOUR CODE HERE
import sys

def solve():
    n = int(sys.stdin.readline())
    adj = [[] for _ in range(n)]
    total_weight = 0
    for _ in range(n - 1):
        u, v, w = map(int, sys.stdin.readline().split())
        u -= 1
        v -= 1
        adj[u].append((v, w))
        adj[v].append((u, w))
        total_weight += w

    leaves = 0
    for i in range(n):
        if len(adj[i]) == 1:
            leaves += 1

    max_path = 0
    
    def dfs(u, p, current_path):
        nonlocal max_path
        max_branch = 0
        for v, w in adj[u]:
            if v != p:
                branch_len = dfs(v, u, current_path + w)
                max_branch = max(max_branch, branch_len + w)
        
        max_path = max(max_path, current_path + max_branch)
        return max_branch
        
    dfs(0,-1,0)
    
    print(total_weight * 2 - max_path)


solve()