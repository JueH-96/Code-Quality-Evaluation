def main():
    import sys, math
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    results = []
    # In an infinite binary tree (defined by the rule: for i>=2, parent = i//2)
    # the vertices in the subtree of a node v at exactly d levels below v are exactly
    # the nodes in the interval [v*2^d, v*2^d + 2^d - 1].
    # However, since our tree is only defined for 1 <= node <= N,
    # when counting nodes at level d below a vertex v, we must clip the interval to N.
    #
    # count_layer(v, d, N) returns the number of nodes at exactly d levels below v
    # (i.e. distance = d from v) that are within [1, N].
    def count_layer(v, d, N):
        if d < 0:
            return 0
        lower = v << d  # same as v * (2**d)
        upper = (v << d) + (1 << d) - 1  # v*2^d + 2^d - 1
        if lower > N:
            return 0
        return max(0, min(N, upper) - lower + 1)

    # For each test case, we want to count the number of vertices Y with distance d(X, Y) = K.
    # Notice that in a tree, the unique simple path from X to Y goes upward some steps (say p)
    # from X to an ancestor A and then downward (d = K - p) from A to Y.
    # For a fixed upward count p, A = ancestor obtained by shifting X right by p, i.e., A = X >> p.
    # However, if p > 0 the downward path from A must not go into the branch that leads
    # back to X because that would be counted by a different decomposition (the shorter upward route).
    #
    # For p = 0 (i.e. A = X), we simply count the nodes exactly d levels below X.
    # For p > 0:
    #   We count nodes at distance d from A (i.e. in the interval [A*2^d, A*2^d + 2^d - 1])
    #   but subtract the nodes from the branch that is in the direction of X.
    #   That branch is the subtree of the child of A that lies on the path from A to X.
    #   That child is: forbidden = X >> (p - 1); and the nodes from forbidden
    #   that are exactly (d-1) levels below forbidden are exactly those in the forbidden branch.
    
    # Let depth(v) = floor(log2(v)), so that v has that many ancestors (excluding v itself).
    #
    # We sum over all possible p in [0, min(K, depth(X))].
    # For each p, let d = K - p.
    #   If p == 0, answer contribution = count_layer(X, d, N).
    #   For p > 0:
    #       If d == 0, the candidate is just the ancestor A.
    #       If d >= 1, we add: count_layer(A, d, N) - count_layer(forbidden, d - 1, N),
    #       where forbidden = X >> (p - 1).
    #
    # Note: X.bit_length()-1 is floor(log2(X))
    
    pos = 1
    for _ in range(t):
        N = int(data[pos]); pos += 1
        X = int(data[pos]); pos += 1
        K = int(data[pos]); pos += 1
        
        # Special case: if K == 0, then only X has distance 0.
        if K == 0:
            results.append("1")
            continue

        depthX = X.bit_length() - 1  # floor(log2(X))
        total = 0
        max_p = min(K, depthX)
        for p in range(max_p + 1):
            d = K - p  # number of downward moves from ancestor A
            A = X >> p  # ancestor reached by p upward moves from X
            if A < 1 or A > N:
                continue
            if p == 0:
                total += count_layer(A, d, N)
            else:
                if d == 0:
                    total += 1
                else:
                    forbidden = X >> (p - 1)
                    total += count_layer(A, d, N) - count_layer(forbidden, d - 1, N)
        results.append(str(total))
    sys.stdout.write("
".join(results))
    
if __name__ == '__main__':
    main()