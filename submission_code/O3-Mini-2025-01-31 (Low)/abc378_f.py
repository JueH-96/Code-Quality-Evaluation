def main():
    import sys
    sys.setrecursionlimit(3000000)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    graph = [[] for _ in range(n+1)]
    deg = [0]*(n+1)
    for _ in range(n-1):
        u = int(next(it))
        v = int(next(it))
        graph[u].append(v)
        graph[v].append(u)
        deg[u] += 1
        deg[v] += 1

    # Explanation:
    # We have a tree and we want to add one edge so that the unique cycle has
    # the property that every cycle vertex (i.e. every vertex on the cycle formed
    # as the unique path between the endpoints plus the added edge) has degree 3 in
    # the new graph.
    #
    # Let the endpoints of the added edge be u and v.
    # In the tree, they have degrees deg[u] and deg[v]. When we add the edge, these
    # endpoints have degree deg[u] + 1 and deg[v] + 1. Meanwhile every other vertex
    # on the path (the “interior” vertices on the unique u-to-v tree path) retain their tree degree.
    #
    # To require that every vertex on the cycle has degree exactly 3 in the new graph,
    # we must have for endpoints: deg[u] + 1 = 3  => deg[u] = 2, and likewise deg[v] = 2.
    # For every interior vertex x on the u-to-v tree path we must have: deg[x] = 3.
    #
    # Also note that u and v cannot be adjacent in the tree; otherwise the added edge
    # would duplicate an existing edge.
    #
    # So the candidate added edge must connect two vertices u and v,
    # where u and v are both degree 2 in the tree, and the unique tree path
    # between them (which must have at least one interior vertex) consists solely of vertices of degree 3.
    #
    # Our strategy is: for every vertex u with degree 2, use it as one endpoint and explore
    # “chains” from u that go into vertices with degree 3 and eventually end at a vertex of 
    # degree 2. In each such chain u--...--v, the interior nodes will (by construction) all have degree 3.
    # Note that if a chain u--...--v is found, the same chain will be discovered from v as well.
    # Hence we will divide by 2 at the end.
    #
    # In the DFS we work as follows:
    #   From a starting degree-2 vertex u, consider a neighbor v that is degree 3.
    #   Then, from v (with u as “parent”), use DFS to extend the path. At a degree-3 node,
    #   for every neighbor (except the node we came from), if it has degree 3 we continue the chain;
    #   if it has degree 2 then we treat that as the other endpoint of a valid chain.
    def dfs(curr, prev):
        total = 0
        for nxt in graph[curr]:
            if nxt == prev:
                continue
            if deg[nxt] == 2:  # Found termination
                total += 1
            elif deg[nxt] == 3:
                total += dfs(nxt, curr)
            # Other vertices cannot appear by our condition.
        return total

    total_paths = 0
    # For every degree 2 vertex, start DFS along its neighbors that are degree 3.
    for u in range(1, n+1):
        if deg[u] == 2:
            for nbr in graph[u]:
                if deg[nbr] == 3:
                    total_paths += dfs(nbr, u)
    # Each chain is counted twice (once from each endpoint), so divide by 2.
    print(total_paths // 2)

if __name__ == '__main__':
    main()