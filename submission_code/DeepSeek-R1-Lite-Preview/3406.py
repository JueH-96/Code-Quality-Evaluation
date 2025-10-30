class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dp(last, i, j, run_length):
            if i == zero and j == one:
                return 1
            total = 0
            if last == 0:
                # Place a block of 1s
                for run in range(1, limit + 1):
                    if j + run > one:
                        break
                    total += dp(1, i, j + run, 1)
                    total %= MOD
            elif last == 1:
                # Place a block of 0s
                for run in range(1, limit + 1):
                    if i + run > zero:
                        break
                    total += dp(0, i + run, j, 1)
                    total %= MOD
            else:
                # No elements placed yet, can start with 0 or 1
                # Place a block of 0s
                for run in range(1, limit + 1):
                    if i + run > zero:
                        break
                    total += dp(0, i + run, j, run)
                    total %= MOD
                # Place a block of 1s
                for run in range(1, limit + 1):
                    if j + run > one:
                        break
                    total += dp(1, i, j + run, run)
                    total %= MOD
            return total % MOD

        return dp(-1, 0, 0, 0)