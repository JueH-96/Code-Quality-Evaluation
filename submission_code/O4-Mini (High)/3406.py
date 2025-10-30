class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        Z, O, K = zero, one, limit
        MOD = 10**9 + 7

        # dp0[z][o]: # of stable arrays using z zeros and o ones, ending with a 0-run
        # dp1[z][o]: # of stable arrays using z zeros and o ones, ending with a 1-run
        dp0 = [[0] * (O+1) for _ in range(Z+1)]
        dp1 = [[0] * (O+1) for _ in range(Z+1)]
        # P0[z][o] = prefix sum of dp0[z][0..o]
        # P1[z][o] = prefix sum of dp1[0..z][o]
        P0 = [[0] * (O+1) for _ in range(Z+1)]
        P1 = [[0] * (O+1) for _ in range(Z+1)]

        for z in range(Z+1):
            for o in range(O+1):
                if z == 0 and o == 0:
                    # empty prefix: no runs
                    P0[0][0] = 0
                    P1[0][0] = 0
                    continue

                # Compute dp0[z][o]
                if z > 0:
                    if o == 0:
                        # only zeros, valid iff z <= K
                        dp0[z][0] = 1 if z <= K else 0
                    else:
                        # sum of dp1[z-k][o] for k=1..min(K,z)
                        # = P1[z-1][o] - P1[z-K-1][o]  (if z-K-1 >= 0)
                        res = P1[z-1][o]
                        if z-K-1 >= 0:
                            res -= P1[z-K-1][o]
                        dp0[z][o] = res % MOD

                # Compute dp1[z][o]
                if o > 0:
                    if z == 0:
                        # only ones, valid iff o <= K
                        dp1[0][o] = 1 if o <= K else 0
                    else:
                        # sum of dp0[z][o-k] for k=1..min(K,o)
                        # = P0[z][o-1] - P0[z][o-K-1]  (if o-K-1 >= 0)
                        res = P0[z][o-1]
                        if o-K-1 >= 0:
                            res -= P0[z][o-K-1]
                        dp1[z][o] = res % MOD

                # Update prefix sums
                P0[z][o] = (dp0[z][o] + (P0[z][o-1] if o > 0 else 0)) % MOD
                P1[z][o] = (dp1[z][o] + (P1[z-1][o] if z > 0 else 0)) % MOD

        # The total is the sum of ways ending in a 0-run or a 1-run
        return (dp0[Z][O] + dp1[Z][O]) % MOD