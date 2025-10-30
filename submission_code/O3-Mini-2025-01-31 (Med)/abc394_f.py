def main():
    import sys
    from collections import deque
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    
    # Build the tree (1-indexed)
    graph = [[] for _ in range(n+1)]
    degree = [0]*(n+1)
    for _ in range(n-1):
        u = int(next(it))
        v = int(next(it))
        graph[u].append(v)
        graph[v].append(u)
        degree[u] += 1
        degree[v] += 1

    # In an alkane subgraph every vertex (internal) must have degree exactly 4 in the subgraph.
    # If a vertex is used as an internal vertex then it must end up with 4 incident edges.
    # But if a vertex v is chosen as an internal vertex, the subgraph uses some of the actual
    # edges from T: if d_I(v) is the number of edges connecting v to other internal vertices,
    # then the remaining 4 - d_I(v) edges must come from leaves which are chosen among v's neighbors
    # outside I. For this to be possible we must have:
    #
    #   external count = d_T(v) - d_I(v) >= 4 - d_I(v)
    #
    # which simplifies to d_T(v) >= 4.
    #
    # Thus every vertex chosen as an internal vertex must have degree at least 4 in T.
    # And an alkane subgraph is completely determined by:
    #   • A connected set I of vertices (in T) such that every vertex v in I has degree >= 4,
    #     and such that for each v the number of leaves needed (4 - d_I(v)) can be obtained.
    #   • One can always pick the required leaves because the available neighbors 
    #     outside I are exactly d_T(v) - d_I(v) and d_T(v) >= 4.
    #
    # Moreover, if I has |I| = k >= 1, the tree we build has:
    #   Total vertices = k (internal) + (sum_{v in I} (4 - d_I(v))).
    # It can be shown that in any tree on internal vertices with each internal of degree 4 in the final alkane,
    # one must have sum_{v in I}(4 - d_I(v)) = 2x+2, so the total is 3x+2.
    #
    # So to maximize the number of vertices, we need to maximize the number of internal vertices.
    # And the best we can do is to choose I to be a connected set (subtree) among those vertices v with degree>=4.
    #
    # Thus the answer is 3*(size of largest connected component of vertices with degree>=4)+2,
    # provided there is at least one vertex with degree>=4. Otherwise no alkane subgraph exists.
    
    eligible = [False]*(n+1)
    for i in range(1, n+1):
        if degree[i] >= 4:
            eligible[i] = True

    visited = [False]*(n+1)
    max_component = 0
    for i in range(1, n+1):
        if eligible[i] and not visited[i]:
            count = 0
            dq = deque([i])
            visited[i] = True
            while dq:
                cur = dq.popleft()
                count += 1
                for nx in graph[cur]:
                    if eligible[nx] and not visited[nx]:
                        visited[nx] = True
                        dq.append(nx)
            if count > max_component:
                max_component = count

    if max_component == 0:
        sys.stdout.write("-1")
    else:
        # For a connected internal set of size k, the alkane has 3k+2 vertices.
        ans = 3 * max_component + 2
        sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()