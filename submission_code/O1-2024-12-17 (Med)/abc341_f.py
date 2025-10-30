def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parsing input
    N = int(input_data[0])
    M = int(input_data[1])
    idx = 2
    edges = []
    for _ in range(M):
        u = int(input_data[idx]) - 1
        v = int(input_data[idx+1]) - 1
        idx += 2
        edges.append((u, v))
    W = list(map(int, input_data[idx:idx+N]))
    idx += N
    A = list(map(int, input_data[idx:idx+N]))
    idx += N
    
    # Build adjacency: for each vertex x, we keep only neighbors y
    # where W[y] < W[x], because only those can receive a piece from x.
    adj = [[] for _ in range(N)]
    for (u, v) in edges:
        if W[u] < W[v]:
            adj[v].append(u)
        if W[v] < W[u]:
            adj[u].append(v)
    
    # Group vertices by their weight
    maxW = max(W) if N > 0 else 0
    groups = [[] for _ in range(maxW+1)]
    for x in range(N):
        groups[W[x]].append(x)
    
    # f[x] = maximum number of remove-operations obtainable from a single piece starting on x
    f = [0]*N
    
    # We will process weights in ascending order so that
    # for a vertex x of weight w, all neighbors y with W[y] < w already have f[y] computed.
    
    # Preallocate a dp array for knapsack up to capacity w-1 (size w).
    # We'll reuse it for each vertex but re-initialize as needed.
    
    from math import inf
    
    for w in range(1, maxW+1):
        # Process all vertices of weight w
        for x in groups[w]:
            # If x has no valid neighbors, just f[x] = 1
            if not adj[x]:
                f[x] = 1
                continue
            
            # Quick check: if sum of neighbor-weights < w, we can pick them all
            sum_weights = sum(W[y] for y in adj[x])
            if sum_weights < w:
                # We can include all neighbors
                f[x] = 1 + sum(f[y] for y in adj[x])
                continue
            
            # Otherwise, do a 0-1 knapsack where each neighbor y is an item:
            # cost = W[y], value = f[y], capacity = w-1
            dpvalid = [False]*w
            dp = [0]*w
            dpvalid[0] = True
            
            for y in adj[x]:
                cost = W[y]
                val = f[y]
                # Loop backwards over capacity
                # We only go up to w-1 as the max feasible sum of weights
                for cap in range(w-1, cost-1, -1):
                    if dpvalid[cap - cost]:
                        new_val = dp[cap - cost] + val
                        if not dpvalid[cap] or new_val > dp[cap]:
                            dp[cap] = new_val
                            dpvalid[cap] = True
            
            best_val = 0
            for cap in range(w):
                if dpvalid[cap] and dp[cap] > best_val:
                    best_val = dp[cap]
            
            f[x] = best_val + 1
    
    # Finally, the answer is sum(A[x] * f[x]) for x in [0..N-1].
    # Use a 64-bit-like integer (Python int is unbounded, but we keep track carefully).
    answer = 0
    for x in range(N):
        # f[x] could be quite large; A[x] can be up to 1e9
        answer += A[x] * f[x]
    
    print(answer)

# Call main() at the end
if __name__ == "__main__":
    main()