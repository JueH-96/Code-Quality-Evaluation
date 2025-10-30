class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def f(s):
            digits = [int(c) for c in s]
            n = len(digits)
            
            from functools import lru_cache
            
            @lru_cache(maxsize=None)
            def dp(pos, tight, sum_so_far):
                if pos == n:
                    return 1 if (min_sum <= sum_so_far <= max_sum) else 0
                limit = digits[pos] if tight else 9
                total = 0
                for d in range(0, limit + 1):
                    new_tight = tight and (d == limit)
                    new_sum = sum_so_far + d
                    total = (total + dp(pos + 1, new_tight, new_sum)) % MOD
                return total % MOD
            
            return dp(0, True, 0) % MOD
        
        num1_int = int(num1) - 1
        num1_minus_1 = str(num1_int)
        count_num2 = f(num2)
        count_num1_minus_1 = f(num1_minus_1)
        
        return (count_num2 - count_num1_minus_1) % MOD