class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        dp = [[[-1 for _ in range(max_sum + 1)] for _ in range(2)] for _ in range(len(num2) + 1)]
        
        def dfs(index, is_less, current_sum):
            if index == len(num2):
                return 1 if min_sum <= current_sum <= max_sum else 0
            
            if dp[index][is_less][current_sum] != -1:
                return dp[index][is_less][current_sum]
            
            total = 0
            start = 0 if is_less else int(num2[index])
            for digit in range(start, 10):
                if min_sum <= current_sum + digit <= max_sum:
                    if is_less or digit < int(num2[index]):
                        total += dfs(index + 1, 1, current_sum + digit)
                    else:
                        total += dfs(index + 1, 0, current_sum + digit)
            
            dp[index][is_less][current_sum] = total % MOD
            return dp[index][is_less][current_sum]
        
        result = dfs(0, 0, 0)
        
        dp = [[[-1 for _ in range(max_sum + 1)] for _ in range(2)] for _ in range(len(num1) + 1)]
        
        result -= dfs(0, 0, 0)
        result = (result + MOD) % MOD
        
        return result