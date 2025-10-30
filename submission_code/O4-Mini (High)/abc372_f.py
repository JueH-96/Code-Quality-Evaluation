import sys
def main():
    import sys
    from bisect import bisect_left
    from array import array
    input = sys.stdin.readline
    MOD = 998244353

    N, M, K = map(int, input().split())
    X = [0]*M
    Y = [0]*M
    for i in range(M):
        xi, yi = map(int, input().split())
        X[i], Y[i] = xi, yi

    # Build list of "tail" positions (where jumps originate), always include start = 1
    S_set = {1}
    for xi in X:
        S_set.add(xi)
    S_list = sorted(S_set)
    P = len(S_list)
    pos2idx = {v:i for i, v in enumerate(S_list)}

    # Build condensed graph on these P nodes
    # edges[u] = list of (v, cost) where cost = number of steps (shifts+jump) to go from S[u] to S[v]
    edges = [[] for _ in range(P)]
    # segment edges: shift along cycle from one tail to next tail in S_list
    for i in range(P):
        u = i
        v = (i+1) % P
        # cost of pure shifts from S_list[u] to S_list[v]
        if v > u:
            L = S_list[v] - S_list[u]
        else:
            # wrap around
            L = (S_list[0] + N) - S_list[u]
        # L >= 1
        edges[u].append((v, L))
    # jump edges: from each extra edge X->Y, jump costs 1, then pure shifts until next tail
    for xi, yi in zip(X, Y):
        u = pos2idx[xi]
        # find the next tail >= yi in cycle order
        idx = bisect_left(S_list, yi)
        if idx < P:
            v = idx
            shift_cost = S_list[v] - yi
        else:
            v = 0
            shift_cost = (S_list[0] + N) - yi
        cost = 1 + shift_cost
        edges[u].append((v, cost))

    # dp arrays: dp[u][t] = number of ways to be at tail-state u at time t
    # We'll track only these P*NODES states, not the interior vertices.
    # We also accumulate ans_inter for paths that end inside an edge (interior vertices).
    dp = [ array('I', [0]) * (K+1) for _ in range(P) ]
    start_idx = pos2idx[1]
    dp[start_idx][0] = 1

    ans_inter = 0
    # Main DP: for t from 0..K, for each state u, propagate along edges
    for t in range(K+1):
        for u in range(P):
            val = dp[u][t]
            if not val:
                continue
            # propagate this val along each outgoing edge
            eu = edges[u]
            # inline locals
            for v, w in eu:
                t2 = t + w
                if t2 <= K:
                    dv = dp[v]
                    nv = dv[t2] + val
                    # reduce mod
                    if nv >= MOD:
                        nv -= MOD
                    dv[t2] = nv
                else:
                    # partial traversal: we end in interior vertex between u->v
                    # if t < K < t+w, we add val to ans_inter
                    # (t < K always here if t <= K, and since t+w > K)
                    ans_inter += val
    # sum ways that ended exactly at a tail at time K, plus interior endings
    total = ans_inter
    for u in range(P):
        total += dp[u][K]
    # final mod
    total %= MOD
    print(total)

if __name__ == "__main__":
    main()