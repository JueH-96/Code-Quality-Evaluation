class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        """
        We need to count the number of binary arrays with exactly 'zero' 0s and 'one' 1s,
        such that no subarray of length > limit is all 0s or all 1s. Equivalently, 
        we cannot have more than 'limit' consecutive 0s or 1s.

        That is the same as counting all sequences of length (zero + one) with exactly 
        'zero' zeros and 'one' ones, subject to the constraint that the run of identical 
        bits cannot exceed 'limit'.

        We'll use dynamic programming to count these valid sequences.

        Let dp0[i][j][r] be the number of ways to form a partial sequence using i zeros and j ones
        so far, where the last placed bit is 0, and the current run of 0s has length r (1 <= r <= limit).
        Similarly, dp1[i][j][r] is defined for the last bit = 1, with a run of 'r' ones.

        We'll also have dp_start[i][j], which is the number of ways to arrive at (i,j) used bits 
        without having placed any bit yet (i.e., starting point). Initially dp_start[0][0] = 1, 
        meaning no bits chosen yet.

        Transitions:
          from dp_start[i][j]:
              - if i < zero: dp0[i+1][j][1] += dp_start[i][j]
              - if j < one:  dp1[i][j+1][1] += dp_start[i][j]

          from dp0[i][j][r] (last bit=0, run of zeros=r):
              - if i < zero and r < limit: dp0[i+1][j][r+1] += dp0[i][j][r]
              - if j < one: dp1[i][j+1][1] += dp0[i][j][r]

          from dp1[i][j][r] (last bit=1, run of ones=r):
              - if j < one and r < limit: dp1[i][j+1][r+1] += dp1[i][j][r]
              - if i < zero: dp0[i+1][j][1] += dp1[i][j][r]

        Finally, the answer is the sum of dp0[zero][one][r] + dp1[zero][one][r] for r=1..limit,
        all taken modulo 10^9+7.
        """
        MOD = 10**9 + 7
        
        # Edge case: if zero=0 or one=0, then the only way is if the array is 
        # all ones or all zeros. But that can't be stable if limit < zero+one,
        # although the problem states zero,one >= 1, so not strictly needed.
        
        # dp_start[i][j]: ways to have placed i zeros, j ones with no "last bit"
        dp_start = [[0]*(one+1) for _ in range(zero+1)]
        dp_start[0][0] = 1
        
        # dp0[i][j][r] = ways with i zeros used, j ones used, last bit=0, run length=r
        dp0 = [[[0]*(limit+1) for _ in range(one+1)] for _ in range(zero+1)]
        # dp1[i][j][r] = ways with i zeros used, j ones used, last bit=1, run length=r
        dp1 = [[[0]*(limit+1) for _ in range(one+1)] for _ in range(zero+1)]
        
        # Build transitions from dp_start
        for i in range(zero+1):
            for j in range(one+1):
                ways = dp_start[i][j]
                if ways == 0:
                    continue
                # place a 0
                if i < zero:
                    dp0[i+1][j][1] = (dp0[i+1][j][1] + ways) % MOD
                # place a 1
                if j < one:
                    dp1[i][j+1][1] = (dp1[i][j+1][1] + ways) % MOD
        
        # Build transitions for dp0 and dp1
        for i in range(zero+1):
            for j in range(one+1):
                for r in range(1, limit+1):
                    # If last bit=0, run length=r
                    ways0 = dp0[i][j][r]
                    if ways0:
                        # we can add another 0 if i < zero and r < limit
                        if i < zero and r < limit:
                            dp0[i+1][j][r+1] = (dp0[i+1][j][r+1] + ways0) % MOD
                        # we can switch to a 1 if j < one
                        if j < one:
                            dp1[i][j+1][1] = (dp1[i][j+1][1] + ways0) % MOD
                    
                    # If last bit=1, run length=r
                    ways1 = dp1[i][j][r]
                    if ways1:
                        # we can add another 1 if j < one and r < limit
                        if j < one and r < limit:
                            dp1[i][j+1][r+1] = (dp1[i][j+1][r+1] + ways1) % MOD
                        # we can switch to a 0 if i < zero
                        if i < zero:
                            dp0[i+1][j][1] = (dp0[i+1][j][1] + ways1) % MOD
        
        # Sum up all valid ways that used exactly zero 0's and one 1's
        result = 0
        for r in range(1, limit+1):
            result = (result + dp0[zero][one][r]) % MOD
            result = (result + dp1[zero][one][r]) % MOD
        
        return result