class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        # dp0[i][j]: number of valid sequences with i zeros and j ones, ending with a block of 0s
        # dp1[i][j]: number of valid sequences with i zeros and j ones, ending with a block of 1s
        dp0 = [[0] * (one + 1) for _ in range(zero + 1)]
        dp1 = [[0] * (one + 1) for _ in range(zero + 1)]
        
        # Seed the base cases: a sequence consisting of only zeros (resp. only ones)
        # of length up to 'limit' is valid (intermediate states, may be extended later).
        for t in range(1, min(zero, limit) + 1):
            dp0[t][0] = 1
        for t in range(1, min(one, limit) + 1):
            dp1[0][t] = 1
        
        # Fill DP in increasing order of (i+j)
        for i in range(zero + 1):
            for j in range(one + 1):
                # Skip the empty-empty state
                if i == 0 and j == 0:
                    continue
                
                # Transition to dp0[i][j]: end with a block of zeros of length 1..limit
                # We must have come from dp1[i - t][j], t zeros appended.
                if j > 0 or (j == 0 and i > 0):
                    total0 = 0
                    max_t = min(i, limit)
                    for t in range(1, max_t + 1):
                        total0 += dp1[i - t][j]
                    dp0[i][j] = (dp0[i][j] + total0) % MOD
                
                # Transition to dp1[i][j]: end with a block of ones of length 1..limit
                # We must have come from dp0[i][j - t], t ones appended.
                if i > 0 or (i == 0 and j > 0):
                    total1 = 0
                    max_t = min(j, limit)
                    for t in range(1, max_t + 1):
                        total1 += dp0[i][j - t]
                    dp1[i][j] = (dp1[i][j] + total1) % MOD
        
        # The answer is the sum of sequences ending in 0 or ending in 1
        return (dp0[zero][one] + dp1[zero][one]) % MOD