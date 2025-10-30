class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        mod = 10**9 + 7
        
        def digit_sum(n: int) -> int:
            return sum(int(d) for d in str(n))
        
        def dfs(index: int, current_sum: int, is_limit: bool, is_num: bool) -> int:
            if index == len(num2):
                return int(is_num and min_sum <= current_sum <= max_sum)
            if not is_num:
                return 0
            if (index, current_sum, is_limit) in memo:
                return memo[(index, current_sum, is_limit)]
            res = 0
            if not is_limit:
                res = dfs(index + 1, current_sum, False, False)
            low = 0 if is_num else 1
            high = int(num2[index]) if is_limit else 9
            for d in range(low, high + 1):
                res += dfs(index + 1, current_sum + d, is_limit and d == high, True)
            memo[(index, current_sum, is_limit)] = res
            return res
        
        memo = {}
        count1 = dfs(0, 0, True, False)
        memo = {}
        count2 = dfs(0, 0, True, False)
        count3 = dfs(0, 0, True, False)
        
        return (count2 - count1 + mod - (digit_sum(int(num1)) < min_sum or digit_sum(int(num1)) > max_sum)) % mod