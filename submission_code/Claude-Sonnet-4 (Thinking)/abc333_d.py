import sys
from collections import defaultdict

def solve():
    n = int(input())
    
    if n == 1:
        print(1)
        return
    
    # Build adjacency list
    adj = defaultdict(list)
    for _ in range(n - 1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    
    # If vertex 1 is already a leaf
    if len(adj[1]) <= 1:
        print(1)
        return
    
    # Root the tree at vertex 1 and compute subtree costs
    def dfs(node, parent):
        children = [child for child in adj[node] if child != parent]
        
        if not children:  # leaf node
            return 1
        
        # Cost to delete this subtree = sum of costs of all child subtrees + 1
        total_cost = 1
        for child in children:
            total_cost += dfs(child, node)
        
        return total_cost
    
    # Calculate cost for each child subtree of vertex 1
    costs = []
    for child in adj[1]:
        cost = dfs(child, 1)
        costs.append(cost)
    
    # To make vertex 1 a leaf, we need to delete all but at most one subtree
    # Choose to keep the most expensive subtree and delete all others
    costs.sort()
    
    # Answer = 1 (to delete vertex 1) + sum of costs of all subtrees except the largest
    if len(costs) == 1:
        answer = 1 + costs[0]
    else:
        answer = 1 + sum(costs[:-1])
    
    print(answer)

solve()