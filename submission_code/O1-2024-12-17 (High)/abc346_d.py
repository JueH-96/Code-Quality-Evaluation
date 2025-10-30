def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S_str = data[1]
    C = list(map(int, data[2:]))

    # Convert string S to array of bits
    S = [int(ch) for ch in S_str]

    # We will build two "alternating" reference patterns:
    #   alt0[i] = i%2, alt1[i] = 1 - (i%2)
    # For each pattern (alt0 or alt1), we consider:
    #   costX_same[i] = cost if we want to make S[i] = altX[i]
    #   costX_flip[i] = cost if we want to make S[i] = (1 - altX[i])
    #
    # A "good string" T must have exactly one pair of adjacent equal bits.
    # Equivalently, it is identical to an all-alternating pattern except
    # for exactly one "skip" in the alternation.  Concretely, T can be:
    #   T^0_j: alt0 up to index j, and then (alt0 ^ 1) from index j+1 onward
    #   T^1_j: alt1 up to index j, and then (alt1 ^ 1) from index j+1 onward
    # for some j in [0..N-2].  Each such T^x_j has exactly one adjacency
    # where T_i = T_{i+1}.
    #
    # We compute the cost to transform S into each T^x_j as:
    #   sum_{i=0..j} costX_same[i] + sum_{i=j+1..N-1} costX_flip[i]
    # We do this for x in {0,1} and j in [0..N-2], taking the minimum.

    cost0_same = [0]*N
    cost0_flip = [0]*N
    cost1_same = [0]*N
    cost1_flip = [0]*N

    for i in range(N):
        # alt0 and alt1
        a0 = i % 2
        a1 = 1 - a0

        # cost to match alt0[i]
        cost0_same[i] = C[i] if S[i] != a0 else 0
        # cost to match (1 - alt0[i])
        cost0_flip[i] = C[i] if S[i] != (1 - a0) else 0

        # cost to match alt1[i]
        cost1_same[i] = C[i] if S[i] != a1 else 0
        # cost to match (1 - alt1[i])
        cost1_flip[i] = C[i] if S[i] != (1 - a1) else 0

    # Prefix sums
    p0s = [0] * (N + 1)  # prefix sums of cost0_same
    p0f = [0] * (N + 1)  # prefix sums of cost0_flip
    p1s = [0] * (N + 1)
    p1f = [0] * (N + 1)

    for i in range(N):
        p0s[i + 1] = p0s[i] + cost0_same[i]
        p0f[i + 1] = p0f[i] + cost0_flip[i]
        p1s[i + 1] = p1s[i] + cost1_same[i]
        p1f[i + 1] = p1f[i] + cost1_flip[i]

    # Compute minimum cost among T^0_j or T^1_j for j in [0..N-2]
    # cost(T^0_j) = p0s[j+1] + (p0f[N] - p0f[j+1])
    # cost(T^1_j) = p1s[j+1] + (p1f[N] - p1f[j+1])

    ans = None
    for j in range(N - 1):
        c0 = p0s[j + 1] + (p0f[N] - p0f[j + 1])
        c1 = p1s[j + 1] + (p1f[N] - p1f[j + 1])
        best = min(c0, c1)
        if ans is None or best < ans:
            ans = best

    print(ans)

# Don't forget to call main()
if __name__ == "__main__":
    main()