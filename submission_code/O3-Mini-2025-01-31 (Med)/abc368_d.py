def main():
    import sys
    sys.setrecursionlimit(300000)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    
    # Build the tree as an adjacency list.
    adj = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u = int(next(it))
        v = int(next(it))
        adj[u].append(v)
        adj[v].append(u)
    
    # Read the required vertices.
    required = set()
    req_list = []
    for _ in range(k):
        v = int(next(it))
        required.add(v)
        req_list.append(v)
    
    # If only one required vertex then the answer is trivially 1.
    if k == 1:
        sys.stdout.write("1")
        return
    
    # Choose any required vertex as the root for DFS.
    root = req_list[0]
    
    # We'll mark vertices that are in the minimal subtree (union of all paths among required vertices).
    included = [False] * (n + 1)
    
    # DFS that returns True if the subtree rooted at u contains any required vertex.
    def dfs(u, parent):
        contains_required = (u in required)
        for v in adj[u]:
            if v == parent:
                continue
            if dfs(v, u):
                contains_required = True
        included[u] = contains_required
        return contains_required
    
    dfs(root, -1)
    
    # Count the vertices included in the minimal subtree.
    result = sum(1 for i in range(1, n + 1) if included[i])
    sys.stdout.write(str(result))


if __name__ == '__main__':
    main()