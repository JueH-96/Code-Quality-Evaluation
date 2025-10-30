class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        upper_bound = int(n**(1/x))
        memo = {}

        def count_ways(remaining_n, current_num):
            if remaining_n == 0:
                return 1
            if remaining_n < 0:
                return 0
            if current_num > upper_bound:
                return 0
            if (remaining_n, current_num) in memo:
                return memo[(remaining_n, current_num)]
            
            power_val = current_num**x
            if power_val > remaining_n:
                result = count_ways(remaining_n, current_num + 1)
            else:
                include_count = count_ways(remaining_n - power_val, current_num + 1)
                exclude_count = count_ways(remaining_n, current_num + 1)
                result = (include_count + exclude_count) % MOD
            memo[(remaining_n, current_num)] = result
            return result

        return count_ways(n, 1)