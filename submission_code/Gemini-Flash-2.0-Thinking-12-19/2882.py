class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        max_num = int(n**(1/x)) + 1
        dp = {}

        def solve(remaining_n, current_num):
            if remaining_n == 0:
                return 1
            if remaining_n < 0:
                return 0
            if current_num > max_num:
                return 0

            if (remaining_n, current_num) in dp:
                return dp[(remaining_n, current_num)]

            power_val = current_num**x
            if power_val > remaining_n:
                res = solve(remaining_n, current_num + 1)
            else:
                include_way = solve(remaining_n - power_val, current_num + 1)
                exclude_way = solve(remaining_n, current_num + 1)
                res = (include_way + exclude_way) % MOD
            
            dp[(remaining_n, current_num)] = res
            return res

        return solve(n, 1)