def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    
    # Build the tree graph and record degree counts.
    graph = [[] for _ in range(n + 1)]
    degree = [0] * (n + 1)
    for _ in range(n - 1):
        u = int(next(it))
        v = int(next(it))
        graph[u].append(v)
        graph[v].append(u)
        degree[u] += 1
        degree[v] += 1
    
    # In any alkane subgraph, the vertices have degree either 1 or 4
    # and the tree structure forces that if we let k be the number of vertices of degree 4,
    # then the number of leaves is 2k + 2 and total vertices = 3k + 2.
    # A key observation is that when we pick a vertex v to be a degree 4 vertex
    # in the subgraph, v must have at least (4 - d_H(v)) neighbors outside the backbone.
    # A short check shows that a vertex from T can serve as a backbone vertex (degree 4 in the subgraph)
    # only if it has degree at least 4 in T.
    #
    # Hence, let B be the set of vertices in T with degree at least 4.
    # Then any valid backbone is a connected subtree H of the induced graph on B.
    # For any connected component H on B, if |H| = k, then we can always “attach”
    # exactly (4 - (backbone degree)) leaves to each vertex (since degree in T is >=4)
    # and obtain an alkane of total size 3k + 2.
    #
    # Thus, we need to find the largest connected component (in T) among vertices with degree>=4.
    
    backbone = [False] * (n + 1)
    for v in range(1, n + 1):
        if degree[v] >= 4:
            backbone[v] = True

    visited = [False] * (n + 1)
    max_component_size = 0

    # Use DFS/BFS to get connected components in the induced subgraph on backbone vertices.
    for v in range(1, n + 1):
        if backbone[v] and not visited[v]:
            stack = [v]
            visited[v] = True
            comp_size = 0
            while stack:
                cur = stack.pop()
                comp_size += 1
                for nei in graph[cur]:
                    if backbone[nei] and not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            max_component_size = max(max_component_size, comp_size)

    # No valid backbone => no alkane exists.
    if max_component_size == 0:
        sys.stdout.write(str(-1))
    else:
        # For a given backbone of size k, the alkane subgraph will have 3*k + 2 vertices.
        result = 3 * max_component_size + 2
        sys.stdout.write(str(result))

if __name__ == '__main__':
    main()