def solve():
    import sys
    sys.setrecursionlimit(10**7)
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    edges = input_data[1:2*(N-1)+1]
    C_vals = input_data[2*(N-1)+1:]

    # Build adjacency list
    graph = [[] for _ in range(N+1)]
    idx = 0
    for _ in range(N-1):
        A = int(edges[idx]); B = int(edges[idx+1])
        idx += 2
        graph[A].append(B)
        graph[B].append(A)
    
    # Read C array, 1-based
    C = [0]*(N+1)
    for i in range(N):
        C[i+1] = int(C_vals[i])
    
    # Step 1: BFS from node 1 to get distances dist[1..N]
    dist = [-1]*(N+1)
    dist[1] = 0
    queue = deque([1])
    while queue:
        node = queue.popleft()
        for nxt in graph[node]:
            if dist[nxt] == -1:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    
    # f(1) = sum(C[i] * dist[i])
    f1 = 0
    for i in range(1, N+1):
        f1 += C[i] * dist[i]
    
    # Step 2: Compute sumCsubtree for each node (root the tree at 1)
    sumCsubtree = [0]*(N+1)
    def dfs_subtree(u, p):
        sumCsubtree[u] = C[u]
        for nxt in graph[u]:
            if nxt == p:
                continue
            dfs_subtree(nxt, u)
            sumCsubtree[u] += sumCsubtree[nxt]
    
    dfs_subtree(1, 0)
    
    # Step 3: We'll use the relation to compute f(x) for all x.
    # f(child) = f(parent) + (S - 2*sumCsubtree[child]) when we go child from parent
    S = sum(C[1:])  # total sum of C
    fval = [0]*(N+1)
    fval[1] = f1
    min_f = f1
    
    def dfs_f(u, p):
        nonlocal min_f
        for nxt in graph[u]:
            if nxt == p:
                continue
            # f(nxt) = f(u) + S - 2*sumCsubtree[nxt]
            fval[nxt] = fval[u] + S - 2*sumCsubtree[nxt]
            if fval[nxt] < min_f:
                min_f = fval[nxt]
            dfs_f(nxt, u)
    
    dfs_f(1, 0)
    
    print(min_f)