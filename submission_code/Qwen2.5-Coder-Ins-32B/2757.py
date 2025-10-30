class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def count_with_limit(limit: str) -> int:
            def dp(pos: int, tight: bool, current_sum: int, memo) -> int:
                if pos == len(limit):
                    return min_sum <= current_sum <= max_sum
                if (pos, tight, current_sum) in memo:
                    return memo[(pos, tight, current_sum)]
                
                result = 0
                upper_bound = int(limit[pos]) if tight else 9
                for digit in range(upper_bound + 1):
                    new_tight = tight and digit == upper_bound
                    result = (result + dp(pos + 1, new_tight, current_sum + digit, memo)) % MOD
                
                memo[(pos, tight, current_sum)] = result
                return result
            
            return dp(0, True, 0, {})

        # Count numbers up to num2 and subtract numbers up to num1-1
        return (count_with_limit(num2) - count_with_limit(str(int(num1) - 1)) + MOD) % MOD