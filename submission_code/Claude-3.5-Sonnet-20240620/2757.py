class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def digit_sum(n):
            return sum(int(d) for d in str(n))

        def count_up_to(num):
            n = len(num)
            dp = [[[0] * (max_sum + 1) for _ in range(2)] for _ in range(n + 1)]
            dp[0][0][0] = 1

            for i in range(1, n + 1):
                for tight in range(2):
                    for s in range(max_sum + 1):
                        if tight:
                            limit = int(num[i - 1])
                        else:
                            limit = 9

                        for d in range(limit + 1):
                            new_s = s + d
                            if new_s > max_sum:
                                break
                            new_tight = tight and d == limit
                            dp[i][new_tight][new_s] = (dp[i][new_tight][new_s] + dp[i-1][tight][s]) % MOD

            return sum(dp[n][0][s] + dp[n][1][s] for s in range(min_sum, max_sum + 1)) % MOD

        return (count_up_to(num2) - count_up_to(str(int(num1) - 1)) + MOD) % MOD