def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    MOD = 998244353

    # Read inputs
    N, M, K = map(int, input_data[:3])
    edges = input_data[3:]

    # If M=0, the graph is just a single directed cycle of length N.
    # From vertex 1, in exactly K steps, there is exactly one way to walk around the cycle
    # (you have no choice at each step). So answer = 1.
    if M == 0:
        # Just print 1 % 998244353
        print(1 % MOD)
        return

    # Build adjacency info only for the chord edges (the cycle edges are implicit).
    # chord_out[x] will hold all destinations from x (0-based) due to chord edges.
    chord_out = [[] for _ in range(N)]
    idx = 0
    for _ in range(M):
        x = int(edges[idx]) - 1
        y = int(edges[idx+1]) - 1
        idx += 2
        chord_out[x].append(y)

    # dp[i] = number of ways to be at vertex i (0-based) after the current number of steps.
    dp = [0]*N
    dp[0] = 1  # start at vertex 0 (which is "vertex 1" in the problem statement)

    # Repeatedly apply the transition K times.
    # Transition:
    #   next_dp[j] = dp[j-1] (cycle edge from j-1 -> j)
    #   plus for each chord edge x->y, add dp[x] to next_dp[y].
    #
    # We do this in O(N + M) per step.  (N updates for the cycle, plus up to M additions
    # for each vertex -- but typically we loop over x in [0..N-1] and do c_x additions,
    # summing c_x=M. So total O(N + M) per iteration.)
    #
    # NOTE: For very large N and K, this can be quite big. However, under the given
    # constraints, the intended (and simplest) solution is this direct simulation.
    #
    # We'll implement it carefully and use modulo at each step.
    
    for _ in range(K):
        next_dp = [0]*N

        # Add contributions from the cycle edges
        # next_dp[(x+1)%N] += dp[x]
        # We do this in one pass:
        for x in range(N):
            nxt = x+1
            if nxt == N:
                nxt = 0
            next_dp[nxt] = (next_dp[nxt] + dp[x]) % MOD

        # Add contributions from chord edges
        # for each x, for each y in chord_out[x]: next_dp[y] += dp[x]
        for x in range(N):
            val = dp[x]
            if val != 0:  # small optimization
                for y in chord_out[x]:
                    next_dp[y] = (next_dp[y] + val) % MOD
        
        dp = next_dp

    # The answer is the sum of dp[i] for i in [0..N-1].
    print(sum(dp) % MOD)

# Don't forget to call main()
if __name__ == "__main__":
    main()