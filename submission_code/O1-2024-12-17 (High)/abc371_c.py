def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    idx = 0

    # Read number of vertices
    N = int(input_data[idx]); idx += 1

    # Precompute all (i<j) pairs and their indices for bitmask representation
    # There will be n_edges = N*(N-1)//2 possible edges.
    pair_idx = {}
    edges = []
    k = 0
    for i in range(N):
        for j in range(i+1, N):
            pair_idx[(i, j)] = k
            edges.append((i, j))
            k += 1
    n_edges = k

    # Read G
    M_G = int(input_data[idx]); idx += 1
    adj_mask_G = 0
    for _ in range(M_G):
        u, v = int(input_data[idx]), int(input_data[idx+1])
        idx += 2
        # convert to 0-based
        u -= 1
        v -= 1
        if u > v:
            u, v = v, u
        adj_mask_G |= (1 << pair_idx[(u, v)])

    # Read H
    M_H = int(input_data[idx]); idx += 1
    adj_mask_H = 0
    for _ in range(M_H):
        a, b = int(input_data[idx]), int(input_data[idx+1])
        idx += 2
        # convert to 0-based
        a -= 1
        b -= 1
        if a > b:
            a, b = b, a
        adj_mask_H |= (1 << pair_idx[(a, b)])

    # Read toggle costs A_{i,j}
    cost_edge = [0]*n_edges
    for i in range(N-1):
        length = N-1 - i
        for x in range(length):
            c = int(input_data[idx]); idx += 1
            j = i + 1 + x
            cost_edge[pair_idx[(i, j)]] = c

    # If N=1, no edges at all, cost is always 0
    if N == 1:
        print(0)
        return

    import itertools
    INF = float('inf')
    best = INF

    # Try all permutations p of {0,1,...,N-1}
    # We want to match G under permutation p with H after toggles
    # That is: G(i,j) should match H'(p[i], p[j]) where H' is H after toggles.
    # Equivalently, we check how G, permuted onto the labeling of H, differs from H.
    # The cost is the sum of toggle costs for each differing edge bit.
    for p in itertools.permutations(range(N)):
        # Build inverse permutation for convenience
        invp = [0]*N
        for i in range(N):
            invp[p[i]] = i

        # Build the adjacency of G in H's labeling according to p
        # permG(x,y) = G(invp[x], invp[y])
        perm_g = 0
        for bitidx, (x, y) in enumerate(edges):
            i, j = invp[x], invp[y]
            if i > j:
                i, j = j, i
            # Check if G has edge (i,j)
            if (adj_mask_G & (1 << pair_idx[(i, j)])) != 0:
                perm_g |= (1 << bitidx)

        # Compare perm_g with H to see which edges differ
        diff = perm_g ^ adj_mask_H

        # Sum up toggle costs for each differing bit
        cost = 0
        for b in range(n_edges):
            if (diff & (1 << b)) != 0:
                cost += cost_edge[b]
                # Early stop if already worse than current best
                if cost >= best:
                    break

        if cost < best:
            best = cost

    print(best)

# Don't forget to call main()!
if __name__ == "__main__":
    main()