def main():
    import sys
    sys.setrecursionlimit(3000000)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    # Build the undirected tree graph as an adjacency list.
    graph = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u = int(next(it))
        v = int(next(it))
        graph[u].append(v)
        graph[v].append(u)
    
    # Mark required vertices.
    required = [False] * (n + 1)
    req_nodes = []
    for _ in range(k):
        v = int(next(it))
        required[v] = True
        req_nodes.append(v)
    
    # The idea is to compute the minimal subtree containing all required nodes.
    # We perform a DFS starting from any required node.
    # During the DFS, we mark a node as "needed" if it is one of the required nodes
    # or if any of its descendants is needed. We then count how many nodes are marked needed.
    ans = 0
    def dfs(u, parent):
        nonlocal ans
        needed = required[u]
        for v in graph[u]:
            if v == parent:
                continue
            if dfs(v, u):
                needed = True
        if needed:
            ans += 1
        return needed

    # Start DFS from any required node.
    start = req_nodes[0]
    dfs(start, -1)
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()