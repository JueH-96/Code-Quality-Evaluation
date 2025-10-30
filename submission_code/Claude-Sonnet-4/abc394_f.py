import sys
from collections import defaultdict

def solve():
    n = int(input())
    adj = defaultdict(list)
    
    for _ in range(n - 1):
        a, b = map(int, input().split())
        adj[a].append(b)
        adj[b].append(a)
    
    max_size = 0
    
    # Try each node as a potential degree-4 node
    for center in range(1, n + 1):
        if len(adj[center]) < 4:
            continue
            
        # DFS to find maximum subtree size from each neighbor
        def dfs(v, parent):
            size = 1
            for neighbor in adj[v]:
                if neighbor != parent:
                    size += dfs(neighbor, v)
            return size
        
        # Get sizes of all subtrees rooted at neighbors of center
        subtree_sizes = []
        for neighbor in adj[center]:
            subtree_sizes.append(dfs(neighbor, center))
        
        # Sort in descending order and take the 4 largest
        subtree_sizes.sort(reverse=True)
        if len(subtree_sizes) >= 4:
            current_size = 1 + sum(subtree_sizes[:4])
            max_size = max(max_size, current_size)
    
    print(max_size if max_size > 0 else -1)

solve()