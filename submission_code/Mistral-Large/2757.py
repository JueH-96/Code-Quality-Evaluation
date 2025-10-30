class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def digit_dp(s, max_sum):
            n = len(s)
            dp = [[0] * (max_sum + 1) for _ in range(n + 1)]
            dp[0][0] = 1

            for i in range(1, n + 1):
                max_digit = int(s[i - 1])
                for sum_val in range(max_sum + 1):
                    for digit in range(max_digit + 1):
                        if sum_val >= digit:
                            dp[i][sum_val] = (dp[i][sum_val] + dp[i - 1][sum_val - digit]) % MOD

            total = sum(dp[n][sum_val] for sum_val in range(min_sum, max_sum + 1)) % MOD
            return total

        def count_in_range(s, min_sum, max_sum):
            n = len(s)
            dp = [[0] * (max_sum + 1) for _ in range(n + 1)]
            dp[0][0] = 1

            for i in range(1, n + 1):
                max_digit = int(s[i - 1])
                for sum_val in range(max_sum + 1):
                    for digit in range(max_digit + 1):
                        if sum_val >= digit:
                            dp[i][sum_val] = (dp[i][sum_val] + dp[i - 1][sum_val - digit]) % MOD
                    if i < n:
                        for digit in range(max_digit + 1, 10):
                            if sum_val >= digit:
                                dp[i][sum_val] = (dp[i][sum_val] + dp[i - 1][sum_val - digit]) % MOD

            total = sum(dp[n][sum_val] for sum_val in range(min_sum, max_sum + 1)) % MOD
            return total

        count1 = count_in_range(num1, min_sum, max_sum)
        count2 = count_in_range(num2, min_sum, max_sum)

        return (count2 - count1 + MOD) % MOD

# Example usage:
# solution = Solution()
# print(solution.count("1", "12", 1, 8))  # Output: 11
# print(solution.count("1", "5", 1, 5))    # Output: 5