# YOUR CODE HERE
n = int(input())
adj = [[] for _ in range(n + 1)]

for _ in range(n - 1):
    u, v = map(int, input().split())
    adj[u].append(v)
    adj[v].append(u)

# If vertex 1 is already a leaf
if len(adj[1]) <= 1:
    print(1)
else:
    # Root the tree at vertex 1
    f = [0] * (n + 1)
    
    def dfs(v, parent):
        children = []
        for u in adj[v]:
            if u != parent:
                dfs(u, v)
                children.append(f[u])
        
        if not children:  # v is a leaf
            f[v] = 1
        else:
            children.sort(reverse=True)
            f[v] = max(children[i] + i for i in range(len(children))) + 1
    
    # Compute f values for all subtrees rooted at neighbors of vertex 1
    neighbors = adj[1]
    neighbor_costs = []
    for neighbor in neighbors:
        dfs(neighbor, 1)
        neighbor_costs.append(f[neighbor])
    
    # Sort neighbor costs in descending order
    neighbor_costs.sort(reverse=True)
    
    # The answer is the cost to delete all but one neighbor, plus 1 to delete vertex 1
    k = len(neighbor_costs)
    ans = max(neighbor_costs[i] + i - 1 for i in range(1, k)) + 1
    print(ans)