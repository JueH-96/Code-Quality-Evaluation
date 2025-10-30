# YOUR CODE HERE
def solve():
    n = int(input())
    
    if n == 1:
        print(1)
        return
    
    # Build adjacency list
    adj = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    
    # DFS to calculate minimum operations needed to delete each vertex
    def dfs(v, parent):
        children = []
        for u in adj[v]:
            if u != parent:
                children.append(dfs(u, v))
        
        if not children:  # v is a leaf
            return 1
        
        # For vertex 1, we need to delete all subtrees
        if v == 1:
            return sum(children)
        
        # For other vertices, we can keep one subtree (the one with max cost)
        # and delete all others, then delete this vertex
        return sum(children) - max(children) + 1
    
    result = dfs(1, -1)
    print(result)

solve()