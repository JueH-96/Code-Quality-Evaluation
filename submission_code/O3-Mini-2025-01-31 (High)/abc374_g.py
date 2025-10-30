def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    
    nVertices = 26
    # Build directed graph (adjacency matrix) – edge X→Y if product name XY is used.
    graph = [[False] * nVertices for _ in range(nVertices)]
    outdeg = [0] * nVertices
    indeg = [0] * nVertices
    for _ in range(N):
        edge = next(it).strip()
        if not edge:
            continue
        u = ord(edge[0]) - ord('A')
        v = ord(edge[1]) - ord('A')
        if not graph[u][v]:
            graph[u][v] = True
            outdeg[u] += 1
            indeg[v] += 1

    # Compute reachability matrix R (26x26):
    R = [[False] * nVertices for _ in range(nVertices)]
    for i in range(nVertices):
        for j in range(nVertices):
            if i == j or graph[i][j]:
                R[i][j] = True
    for k in range(nVertices):
        for i in range(nVertices):
            if R[i][k]:
                for j in range(nVertices):
                    if R[k][j]:
                        R[i][j] = True

    # Determine which vertices appear in some edge.
    used = [False] * nVertices
    for i in range(nVertices):
        if outdeg[i] + indeg[i] > 0:
            used[i] = True

    # Build an undirected graph (for weak connectivity):
    undirected = [[] for _ in range(nVertices)]
    for i in range(nVertices):
        for j in range(nVertices):
            if (graph[i][j] or graph[j][i]) and used[i] and used[j]:
                undirected[i].append(j)
                undirected[j].append(i)
    # DFS to find weakly connected components.
    visited = [False] * nVertices
    components = []
    for i in range(nVertices):
        if used[i] and not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nb in undirected[cur]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(comp)

    # For maximum bipartite matching we use DFS (Kuhn's algorithm).
    def bipartite_match(adj, nL, nR):
        matchR = [-1] * nR
        def dfs(u, visited):
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if matchR[v] == -1 or dfs(matchR[v], visited):
                    matchR[v] = u
                    return True
            return False
        result = 0
        for u in range(nL):
            visitedR = [False] * nR
            if dfs(u, visitedR):
                result += 1
        return result

    totalAns = 0
    # Process each weak component separately.
    for comp in components:
        # Compute imbalance d(v) = outdeg - indeg for v in comp.
        comp_d = {}
        for v in comp:
            comp_d[v] = outdeg[v] - indeg[v]
        pos_sum = 0
        for v in comp:
            if comp_d[v] > 0:
                pos_sum += comp_d[v]
        T0 = pos_sum if pos_sum >= 1 else 1
 
        # Build bipartite graph:
        # Left side: for each v with d(v) < 0, add (-d(v)) copies.
        # Right side: for each u with d(u) > 0, add (d(u)) copies.
        leftList = []   # each element is a vertex (letter index)
        rightList = []
        for v in comp:
            if comp_d[v] < 0:
                for _ in range(-comp_d[v]):
                    leftList.append(v)
            elif comp_d[v] > 0:
                for _ in range(comp_d[v]):
                    rightList.append(v)
        nL = len(leftList)
        nR = len(rightList)
        adj = [[] for _ in range(nL)]
        # For each left copy (vertex v) and each right copy (vertex u),
        # add edge if R[v][u] is True.
        for i in range(nL):
            lv = leftList[i]
            for j in range(nR):
                rv = rightList[j]
                if R[lv][rv]:
                    adj[i].append(j)
 
        M = bipartite_match(adj, nL, nR)
        compAns = T0 - M
        if compAns < 1:
            compAns = 1
        totalAns += compAns
 
    sys.stdout.write(str(totalAns))
 
 
if __name__ == '__main__':
    main()