def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    
    # Build the tree as an adjacency list.
    adj = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        a = int(next(it))
        b = int(next(it))
        adj[a].append(b)
        adj[b].append(a)
        
    # Mark the required vertices.
    is_required = [False] * (n + 1)
    required = []
    for _ in range(k):
        v = int(next(it))
        is_required[v] = True
        required.append(v)
    
    # Choose one of the required vertices as the root.
    # (The tree is connected, so a DFS from any required vertex will visit all nodes.)
    root = required[0] if required else 1
    
    # We define a DFS that returns True if the current node or any of its descendants
    # is a required vertex. We count the node if it lies on any path connecting required vertices.
    # This DFS will mark exactly the nodes in the minimal subtree connecting all required nodes.
    ans = [0]  # Using a list so we can modify it in the nested function.
    
    def dfs(v, parent):
        flag = is_required[v]
        for nxt in adj[v]:
            if nxt == parent:
                continue
            if dfs(nxt, v):
                flag = True
        if flag:
            ans[0] += 1
        return flag
    
    dfs(root, -1)
    sys.stdout.write(str(ans[0]))
    
if __name__ == "__main__":
    main()