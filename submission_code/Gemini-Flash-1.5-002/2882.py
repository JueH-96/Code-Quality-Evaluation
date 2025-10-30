class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        dp = {}

        def countWays(current_sum, current_num):
            if current_sum == n:
                return 1
            if current_sum > n or current_num**x > n:
                return 0
            if (current_sum, current_num) in dp:
                return dp[(current_sum, current_num)]

            ways = countWays(current_sum + current_num**x, current_num + 1)
            ways = (ways + countWays(current_sum, current_num + 1)) % MOD
            dp[(current_sum, current_num)] = ways
            return ways

        return countWays(0, 1)