def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    # For each roulette, we store a triple: (cost, P, outcomes list)
    wheels = []
    for _ in range(N):
        cost = float(next(it))
        P = int(next(it))
        outcomes = [int(next(it)) for _ in range(P)]
        wheels.append((cost, P, outcomes))
    
    # Our goal is to compute dp[x] = minimum expected additional money needed
    # given that we have already accumulated x points (0 <= x < M).
    # dp[M] is 0 (absorbing state).
    #
    # When playing a wheel i, the outcomes are:
    #    With probability (# outcomes that are 0)/P, the next state remains x.
    #    For each outcome s > 0:
    #         if x+s >= M then we succeed (cost 0 extra) because dp[M]=0,
    #         otherwise we must pay dp[x+s] extra.
    #
    # If we denote for wheel i:
    #    cost = C_i, P = number of outcomes.
    #    Let zeros = count of outcomes with S == 0.
    #    Let add = sum( dp[x+s] for outcomes with s>0 and x+s < M ) 
    #          (and if x+s >= M, dp is 0 so we add nothing).
    #
    # Then the expected extra cost from state x if we use wheel i is initially computed as:
    #    E = C_i + (zeros/P)*dp[x] + (1/P)*[sum_{s>0, x+s<M} dp[x+s]]
    #
    # But notice that the unknown dp[x] appears on the right side if zeros > 0.
    # Rearranging,
    #    dp[x] - (zeros/P)*dp[x] = C_i + (1/P)*[sum_{s>0, x+s<M} dp[x+s]]
    # so
    #    dp[x] = ( C_i + (1/P)*[sum_{s>0, x+s<M} dp[x+s]] ) / (1 - zeros/P)
    #
    # Since we have a choice over wheels, the Bellman equation is:
    #    dp[x] = min_{i = 1..N} { ( C_i + (1/P_i) * Σ_{s in outcomes with s>0 and x+s<M} dp[x+s] ) / (1 - (number of outcomes with s==0)/P_i) }.
    #
    # Because every outcome with s>0 gives x+s > x (since s>=1), we can compute dp[x]
    # by proceeding backwards from x = M-1 down to 0.
    
    dp = [0.0] * (M + 1)
    dp[M] = 0.0  # Terminal state (at least M points) has cost 0
    
    # For x = M-1 down to 0, compute dp[x]
    for x in range(M - 1, -1, -1):
        best = float('inf')
        for (cost, P, outcomes) in wheels:
            zeros = 0
            add = 0.0
            for s in outcomes:
                if s == 0:
                    zeros += 1
                else:
                    if x + s < M:
                        add += dp[x + s]
                    # if x+s >= M, then we have reached M so add nothing (dp[M]=0)
            # Since the problem guarantees at least one outcome with s > 0,
            # zeros < P, so (1 - zeros/P) > 0.
            cand = (cost + (add / P)) / (1 - (zeros / P))
            if cand < best:
                best = cand
        dp[x] = best
    
    # Print dp[0] as the expected cost from 0 points.
    sys.stdout.write(f"{dp[0]:.12f}")

if __name__ == '__main__':
    main()