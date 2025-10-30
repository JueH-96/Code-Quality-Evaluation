class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        """
        We want to count the number of binary arrays of total length (zero + one),
        containing exactly 'zero' zeroes and 'one' ones, such that:

        1. The count of 0's is exactly zero.
        2. The count of 1's is exactly one.
        3. No subarray of length > limit is all 0's or all 1's. Equivalently,
           the run (consecutive occurrence) of 0's or 1's must never exceed 'limit'.

        A dynamic programming approach:

        Let dp0[i0][i1][r] = number of ways to form a partial array
          that has used i0 zeroes and i1 ones so far,
          with the last placed bit = 0
          and the current run of 0's = r.
        Similarly dp1[i0][i1][r] for last placed bit = 1 with a run of 1's = r.

        Transitions:
        - From dp0[i0][i1][r], we can:
          * place another 0 (if i0 < zero and r < limit) => dp0[i0+1][i1][r+1]
          * place a 1 (if i1 < one) => dp1[i0][i1+1][1]
        - From dp1[i0][i1][r], we can:
          * place another 1 (if i1 < one and r < limit) => dp1[i0][i1+1][r+1]
          * place a 0 (if i0 < zero) => dp0[i0+1][i1][1]

        We'll start by placing either a 0 (if zero > 0) or a 1 (if one > 0),
        correspondingly setting dp0[1][0][1] or dp1[0][1][1] to 1 for the base case.

        Our final answer is the sum of dp0[zero][one][r] and dp1[zero][one][r]
        over r=1..limit, modulo 10^9+7.
        """
        MOD = 10**9 + 7

        # Edge case: if zero=0 or one=0 not possible in the problem constraints (contradiction),
        # but problem states zero, one >= 1 => no need to handle zero=0 or one=0
        # However, we proceed with general correctness.

        # dp0[i][j][r] = number of ways with i used zeros, j used ones,
        #  last placed bit = 0, run of 0's = r
        dp0 = [[[0]*(limit+1) for _ in range(one+1)] for _ in range(zero+1)]
        # dp1[i][j][r] = number of ways with i used zeros, j used ones,
        #  last placed bit = 1, run of 1's = r
        dp1 = [[[0]*(limit+1) for _ in range(one+1)] for _ in range(zero+1)]

        # Base cases: we can start with a 0 (if zero>0) or a 1 (if one>0)
        if zero > 0:
            dp0[1][0][1] = 1
        if one > 0:
            dp1[0][1][1] = 1

        for i0 in range(zero+1):
            for i1 in range(one+1):
                # transitions from dp0[i0][i1][r]
                for r in range(1, limit+1):
                    count0 = dp0[i0][i1][r]
                    if count0:
                        # place another 0 -> must not exceed the limit in run
                        if i0 < zero and r < limit:
                            dp0[i0+1][i1][r+1] = (dp0[i0+1][i1][r+1] + count0) % MOD
                        # place a 1 -> reset run for 1 to 1
                        if i1 < one:
                            dp1[i0][i1+1][1] = (dp1[i0][i1+1][1] + count0) % MOD

                # transitions from dp1[i0][i1][r]
                for r in range(1, limit+1):
                    count1 = dp1[i0][i1][r]
                    if count1:
                        # place another 1 -> must not exceed the limit in run
                        if i1 < one and r < limit:
                            dp1[i0][i1+1][r+1] = (dp1[i0][i1+1][r+1] + count1) % MOD
                        # place a 0 -> reset run for 0 to 1
                        if i0 < zero:
                            dp0[i0+1][i1][1] = (dp0[i0+1][i1][1] + count1) % MOD

        # Sum up all ways that used exactly 'zero' 0's and 'one' 1's
        ans = 0
        for r in range(1, limit+1):
            ans = (ans + dp0[zero][one][r] + dp1[zero][one][r]) % MOD

        return ans