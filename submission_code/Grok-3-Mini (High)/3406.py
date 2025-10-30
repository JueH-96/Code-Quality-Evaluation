import math

class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 1000000007
        dp = [[[ [0 for _ in range(limit + 1)] for _ in range(2)] for _ in range(one + 1)] for _ in range(zero + 1)]
        
        for s in range(1, zero + one + 1):
            for z in range(max(0, s - one), min(s, zero) + 1):
                o = s - z
                for last in range(2):
                    for run in range(1, limit + 1):
                        if last == 0:
                            if run >= 2:
                                if z - 1 >= 0:
                                    dp[z][o][0][run] = dp[z - 1][o][0][run - 1]
                            elif run == 1:
                                sum_part = 0
                                if z - 1 >= 0:
                                    sum_part = sum(dp[z - 1][o][1][r] for r in range(1, limit + 1)) % MOD
                                add_part = 1 if z == 1 and o == 0 else 0
                                dp[z][o][0][run] = (sum_part + add_part) % MOD
                        else:  # last == 1
                            if run >= 2:
                                if o - 1 >= 0:
                                    dp[z][o][1][run] = dp[z][o - 1][1][run - 1]
                            elif run == 1:
                                sum_part = 0
                                if o - 1 >= 0:
                                    sum_part = sum(dp[z][o - 1][0][r] for r in range(1, limit + 1)) % MOD
                                add_part = 1 if o == 1 and z == 0 else 0
                                dp[z][o][1][run] = (sum_part + add_part) % MOD
        
        ans = 0
        for last in range(2):
            for run in range(1, limit + 1):
                ans += dp[zero][one][last][run]
                ans %= MOD
        return ans