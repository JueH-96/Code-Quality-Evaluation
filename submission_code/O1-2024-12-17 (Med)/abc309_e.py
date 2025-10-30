def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    # Fast extraction of inputs
    N = int(input_data[0])
    M = int(input_data[1])
    
    # p_i array
    p = [0]*(N+1)
    idx = 2
    for i in range(2, N+1):
        p[i] = int(input_data[idx]); idx += 1
    
    # Build children adjacency
    children = [[] for _ in range(N+1)]
    for i in range(2, N+1):
        parent_i = p[i]
        children[parent_i].append(i)
    
    # We will compute depth[] and a BFS order from node 1
    from collections import deque
    depth = [-1]*(N+1)
    parent = [0]*(N+1)
    depth[1] = 0
    parent[1] = 0  # no parent
    order = []
    q = deque([1])
    visited = [False]*(N+1)
    visited[1] = True
    while q:
        u = q.popleft()
        order.append(u)
        for v in children[u]:
            if not visited[v]:
                visited[v] = True
                depth[v] = depth[u] + 1
                parent[v] = u
                q.append(v)
    
    # Read coverage requests, coverage[u] = -1 means no coverage
    coverage = [-1]*(N+1)
    for _ in range(M):
        x = int(input_data[idx]); idx += 1
        y = int(input_data[idx]); idx += 1
        # Increase coverage of x if depth[x] + y is larger
        val = depth[x] + y
        if val > coverage[x]:
            coverage[x] = val
    
    # Propagate coverage top-down in the BFS order we already computed
    # For each node u, first try to inherit coverage from parent[u] if possible,
    # then pass coverage[u] down to each child.
    for u in order:
        par = parent[u]
        if par != 0 and coverage[par] >= depth[u]:
            if coverage[par] > coverage[u]:
                coverage[u] = coverage[par]
        # Now pass coverage[u] to children if within range
        cov_u = coverage[u]  # save to avoid repeated lookups
        if cov_u >= depth[u]:  # node u itself is covered, can push coverage downward
            for c in children[u]:
                if cov_u >= depth[c]:
                    if cov_u > coverage[c]:
                        coverage[c] = cov_u
    
    # Count how many are covered: coverage[i] >= depth[i] means covered
    ans = 0
    for i in range(1, N+1):
        if coverage[i] >= depth[i]:
            ans += 1
    
    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()