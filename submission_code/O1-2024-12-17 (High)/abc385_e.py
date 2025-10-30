def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    edges = input_data[1:]

    adjacency = [[] for _ in range(N)]
    deg = [0]*N

    idx = 0
    for _ in range(N-1):
        u = int(edges[idx]) - 1
        v = int(edges[idx+1]) - 1
        idx += 2
        adjacency[u].append(v)
        adjacency[v].append(u)
        deg[u] += 1
        deg[v] += 1

    # We will keep track of the largest "snowflake" subtree we can form.
    # The snowflake has 1 "center" node c, x "middle" vertices (neighbors of c),
    # each middle has y leaves. In particular, each neighbor must have degree >= (y+1),
    # and all use the same y.
    #
    # For a chosen c, let B = sorted list of degrees of c's neighbors.
    # If we pick y+1 = d, we must have all neighbors with deg >= d. 
    # The total kept neighbors is suffix_size = number of neighbors m with deg(m) >= d.
    # Then the subtree size = 1 (center) + suffix_size * d.
    # We want to maximize 1 + suffix_size * d over all d >= 2 (since y >= 1 => y+1 >= 2).
    #
    # We'll do this for every node c as center, track the best we can do, 
    # and the answer is N - (that best size).

    gbest = 1  # at least 1, though to form a *valid* snowflake we need more.
               # but we'll track the max we can get; it must be ≥ 3 for any valid y≥1, x≥1.
    
    for c in range(N):
        # Gather degrees of neighbors
        B = [deg[m] for m in adjacency[c]]
        B.sort()  # ascending

        # Prepare unique degrees from largest to smallest
        unique_B = []
        last = -1
        for x in B:
            if x != last:
                unique_B.append(x)
                last = x
        
        local_best = 1  # track best for this center
        M = len(B)      # number of neighbors

        # Traverse distinct degrees in descending order
        for d in reversed(unique_B):
            # count how many neighbors have degree >= d
            pos = bisect.bisect_left(B, d)
            suffix_size = M - pos
            if d >= 2:
                # sub-tree size = 1 (center) + suffix_size * d
                val = 1 + suffix_size * d
                if val > local_best:
                    local_best = val
        
        if local_best > gbest:
            gbest = local_best

    # The answer is how many we must delete to leave that largest snowflake
    print(N - gbest)

# Do not forget to call main!
if __name__ == "__main__":
    main()