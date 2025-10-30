class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def digit_sum(x: int) -> int:
            return sum(int(d) for d in str(x))

        def count_good_integers(num: str) -> int:
            n = len(num)
            dp = [[[0] * (max_sum + 1) for _ in range(2)] for _ in range(n + 1)]
            dp[0][1][0] = 1

            for i in range(n):
                for tight in range(2):
                    for s in range(max_sum + 1):
                        if dp[i][tight][s] == 0:
                            continue
                        limit = int(num[i]) if tight else 9
                        for d in range(limit + 1):
                            new_tight = tight and (d == limit)
                            new_sum = s + d
                            if new_sum <= max_sum:
                                dp[i + 1][new_tight][new_sum] = (dp[i + 1][new_tight][new_sum] + dp[i][tight][s]) % MOD

            total_count = 0
            for s in range(min_sum, max_sum + 1):
                total_count = (total_count + dp[n][0][s] + dp[n][1][s]) % MOD

            return total_count

        def subtract_one(num: str) -> str:
            num = list(num)
            i = len(num) - 1
            while i >= 0 and num[i] == '0':
                num[i] = '9'
                i -= 1
            if i >= 0:
                num[i] = str(int(num[i]) - 1)
            return ''.join(num).lstrip('0') or '0'

        count_num2 = count_good_integers(num2)
        num1_minus_one = subtract_one(num1)
        count_num1_minus_one = count_good_integers(num1_minus_one)

        return (count_num2 - count_num1_minus_one + MOD) % MOD