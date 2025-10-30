def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    edges = input_data[1:]
    
    # Adjacency list (1-based indexing)
    adj = [[] for _ in range(N+1)]
    deg = [0]*(N+1)
    
    idx = 0
    for _ in range(N-1):
        u = int(edges[idx]); v = int(edges[idx+1])
        idx += 2
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1

    # We want to find the largest "snowflake subtree"
    # A snowflake with center c, arms = neighbors of c,
    # each arm a can attach up to (deg[a] - 1) leaves.
    # We pick as many arms as we like that can all share the same y ≥ 1.
    # If we sort L = [deg[a]-1 for a in adj[c]] in descending order,
    # and take the first (i+1) arms, y = L[i] (because all must have >= L[i] leaves),
    # then size = 1 + (i+1)*(1 + L[i]). We maximize over i.
    
    max_snowflake = 0

    # For each node c, compute possible arms
    for c in range(1, N+1):
        # If c has no neighbors or only one neighbor that cannot form y>=1, 
        # the resulting snowflake would be invalid (x>0, y>0).
        # Still, we will try the logic, which will yield 0 if no valid (x,y) is found.
        neighbors_info = [deg[a] - 1 for a in adj[c]]
        # Sort descending
        neighbors_info.sort(reverse=True)
        
        local_max = 0
        for i, val in enumerate(neighbors_info):
            # We need y >= 1
            if val < 1:
                break
            # (i+1) arms, each can have 'val' leaves
            sub_size = 1 + (i+1)*(1 + val)
            if sub_size > local_max:
                local_max = sub_size
        
        if local_max > max_snowflake:
            max_snowflake = local_max

    # The answer is how many vertices we must delete: N - size_of_best_snowflake
    print(N - max_snowflake)

# Do not forget to call main()
if __name__ == '__main__':
    main()