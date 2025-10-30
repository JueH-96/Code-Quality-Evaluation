def main():
    import sys,sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    mod = 998244353
    # Parse input: first two numbers are N and M.
    N = int(data[0])
    M = int(data[1])
    # Next N numbers are the elements of B.
    B = [int(x) for x in data[2:2+N]]
    
    # We will do DP. We keep a state dp[m] where m (1 <= m <= M+1) is 
    # the “current minimum” of the prefix. The state holds two numbers:
    #   dp_ways[m] = number of assignments that yield current min m
    #   dp_rec[m]  = total record count (i.e. f(prefix)) over these assignments.
    #
    # We use m = M+1 as sentinel "infinity" (so that any choice in [1, M]
    # yields a new record).
    size = M + 2  # indices 1 .. M+1 (we ignore index 0)
    dp_ways = [0] * (size)
    dp_rec = [0] * (size)
    dp_ways[M+1] = 1  # initial: no vertex assigned, current min = "infinity"
    dp_rec[M+1] = 0
    
    # Process positions 1..N (B[0] ... B[N-1])
    for i in range(N):
        b = B[i]
        if b != -1:
            # Fixed vertex: value is b.
            new_ways = [0] * (size)
            new_rec  = [0] * (size)
            # For each current state m:
            for m in range(1, M+2):
                w = dp_ways[m]
                if w:
                    r = dp_rec[m]
                    if b < m:
                        # new record low formed:
                        # new state becomes b and each assignment gets an extra 1.
                        new_ways[b] = (new_ways[b] + w) % mod
                        new_rec[b] = (new_rec[b] + r + w) % mod
                    else:
                        # no new record low:
                        new_ways[m] = (new_ways[m] + w) % mod
                        new_rec[m] = (new_rec[m] + r) % mod
            dp_ways, dp_rec = new_ways, new_rec
        else:
            # b == -1, free vertex: we can choose any value v in [1,M].
            # The transition: from a current state with value m:
            #   • if we choose v < m then new state becomes v and we add 1 to the record count.
            #   • if we choose v >= m then state remains m.
            #
            # We now “sum” over the contributions from all states.
            new_ways = [0] * (size)
            new_rec  = [0] * (size)
            # Compute cumulative sums from dp.
            # We want for each new state index v in [1,M]:
            #   record branch: new_ways[v] gets add += sum_{m = v+1}^{M+1} dp_ways[m]
            #                   new_rec[v] gets add += sum_{m = v+1}^{M+1} (dp_rec[m] + dp_ways[m])
            # and for staying branch (state remains m): for each m in [1,M],
            #   new_ways[m] gets add += dp_ways[m]*(M - m + 1)
            #   new_rec[m]  gets add += dp_rec[m]*(M - m + 1)
            #
            L = M + 2 # valid indices: 1 ... M+1 ; we will use one extra cell.
            cumF = [0] * (L + 1)   # will store cumulative sums for dp_ways
            cumGF = [0] * (L + 1)  # will store cumulative sums for (dp_rec + dp_ways)
            # Compute cumulative sums from m=M+1 down to 1:
            for m in range(M+1, 0, -1):
                s = dp_ways[m] + cumF[m+1]
                if s >= mod: 
                    s -= mod
                cumF[m] = s
                cumGF[m] = (dp_rec[m] + dp_ways[m] + cumGF[m+1]) % mod
            # Now update new state:
            for v in range(1, M+1):
                # record branch:
                new_ways[v] = cumF[v+1]  % mod
                new_rec[v] = cumGF[v+1]  % mod
                # staying branch: when the current state is m = v,
                # if we choose a value in [v, M] (M - v + 1 choices) then state remains v.
                factor = M - v + 1
                add_w = (dp_ways[v] * factor) % mod
                add_r = (dp_rec[v] * factor) % mod
                new_ways[v] = (new_ways[v] + add_w) % mod
                new_rec[v] = (new_rec[v] + add_r) % mod
            # new_dp state: note that state M+1 gets 0 (as expected)
            dp_ways, dp_rec = new_ways, new_rec

    # The final answer is the sum (over all states) of the record count so far.
    ans = 0
    for m in range(1, M+2):
        ans = (ans + dp_rec[m]) % mod
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()