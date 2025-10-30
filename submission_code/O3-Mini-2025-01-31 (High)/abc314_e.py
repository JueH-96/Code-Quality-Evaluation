def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    # For each roulette wheel we will store:
    #   cost C, number outcomes P, the list of outcomes, and the number of outcomes that give 0 points.
    wheels = []
    for _ in range(N):
        C = float(next(it))
        P = int(next(it))
        outcomes = []
        n0 = 0
        for _ in range(P):
            s = int(next(it))
            outcomes.append(s)
            if s == 0:
                n0 += 1
        wheels.append((C, P, outcomes, n0))
        
    # Let dp[x] be the expected additional cost when Takahashi already has x points and x < M.
    # We define dp[M] = 0 (absorbing state since reaching M points means no further cost).
    # The Bellman optimality equation is:
    #
    #   dp[x] = min_{i in 1..N} { C_i + (1/P_i) * [sum_{j=1}^{P_i} dp[min(M, x + S_{i,j])] ] }
    #
    # However, note that if S_{i,j} = 0 (which means no progress because x+0 = x),
    # then dp[x] appears on both sides. In fact, for a fixed wheel i we have
    #
    #   Expected cost (action i) = C_i + (1/P_i)[ (n0) * dp[x] + (sum of dp[x+s] for outcomes
    #                                with s > 0 and x+s < M) + (0 for outcomes with x+s >= M) ]
    #
    # Let "tot" be the sum over outcomes s > 0 for which x+s < M.
    # Then the above becomes:
    #
    #   dp[x] = C_i + (n0/P_i)*dp[x] + (tot/P_i)
    # => dp[x] * (1 - n0/P_i) = C_i + tot/P_i
    # => dp[x] = (C_i + tot/P_i) / (1 - n0/P_i)
    #
    # And overall the optimal dp[x] is the minimum (over i) of these expressions.
    #
    # Note: For outcomes with x+s >= M, we have dp[>=M] = 0.
    #
    # We will compute dp[x] for x from M-1 downto 0.
    
    dp = [0.0] * (M + 1)
    dp[M] = 0.0  # When already M or more points, no more cost.
    
    for x in range(M - 1, -1, -1):
        best = float("inf")
        for (C, P, outcomes, n0) in wheels:
            tot = 0.0
            # For each outcome: if outcome s > 0 and x+s < M, add dp[x+s].
            # Outcomes with s==0 would yield x and are taken into account via n0; outcomes where x+s>=M
            # finish the game and contribute 0.
            for s in outcomes:
                if s > 0 and x + s < M:
                    tot += dp[x + s]
            # For this wheel, the expected cost (with the self-dependency removed) is:
            #   candidate = (C + tot/P) / (1 - n0/P)
            # Since at outcomes s == 0 we remain in state x.
            candidate = (C + tot / P) / (1 - (n0 / P))
            if candidate < best:
                best = candidate
        dp[x] = best  # Optimal expected cost from state x.
    
    sys.stdout.write("{:.12f}".format(dp[0]))
    
if __name__ == '__main__':
    main()