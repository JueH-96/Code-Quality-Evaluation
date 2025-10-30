class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        m = len(s)
        # Precompute f[d] = number of steps to reduce d to 1 by x -> popcount(x)
        # f[1] = 0; for d>=2: f[d] = 1 + f[popcount(d)]
        # We don't need f[0] since x>0 always.
        f = [0] * (m + 1)
        f[0] = 10**9  # effectively infinity, never used
        f[1] = 0
        # helper to count bits
        def pc(x):
            return bin(x).count("1")
        for d in range(2, m+1):
            f[d] = 1 + f[pc(d)]
        # goodCounts[c] = True if an integer x>1 with popcount=c satisfies 1 + f[c] <= k
        # That is f[c] <= k-1.  For c=1, f[1]=0 so requires k>=1, which holds since k>=1 by constraints.
        good = [False] * (m + 1)
        for c in range(1, m+1):
            if f[c] <= k-1:
                good[c] = True
        
        # DP over bits of s to count numbers < s
        # dp[i][ones][tight]: after i bits, used 'ones' ones, tight=0/1
        # We'll roll over i to save space
        dp_prev = [[0, 0] for _ in range(m+1)]
        # At start: 0 bits used, 0 ones, tight=1 (we match prefix so far)
        dp_prev[0][1] = 1
        
        for i, ch in enumerate(s):
            bit = int(ch)
            dp_curr = [[0, 0] for _ in range(m+1)]
            for ones in range(0, m+1):
                for tight in (0, 1):
                    cnt = dp_prev[ones][tight]
                    if cnt == 0:
                        continue
                    # Put 0 at this position
                    nt = tight and (bit == 0)
                    dp_curr[ones][nt] = (dp_curr[ones][nt] + cnt) % MOD
                    # Put 1 at this position
                    if tight and bit == 0:
                        # can't put 1 if tight and bit is 0
                        pass
                    else:
                        nt2 = tight and (bit == 1)
                        dp_curr[ones+1][nt2] = (dp_curr[ones+1][nt2] + cnt) % MOD
            dp_prev = dp_curr
        
        # Sum up counts for tight=0 (strictly less than s) and good popcounts
        ans = 0
        for ones in range(1, m+1):  # skip ones=0, that's x=0
            if good[ones]:
                ans = (ans + dp_prev[ones][0]) % MOD
        return ans