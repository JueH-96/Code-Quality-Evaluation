class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        mod = 10**9 + 7
        
        def f(s):
            n = len(s)
            from functools import lru_cache
            @lru_cache(maxsize=None)
            def dp(pos, tight, curr_sum):
                if curr_sum > max_sum:
                    return 0
                remaining = n - pos
                if curr_sum + 9 * remaining < min_sum:
                    return 0
                if pos == n:
                    return 1 if min_sum <= curr_sum <= max_sum else 0
                total = 0
                limit = int(s[pos]) if tight else 9
                for d in range(0, limit + 1):
                    new_tight = tight and (d == limit)
                    new_sum = curr_sum + d
                    total = (total + dp(pos + 1, new_tight, new_sum)) % mod
                return total % mod
            
            return dp(0, True, 0)
        
        num1_minus = str(int(num1) - 1)
        total2 = f(num2)
        total1 = f(num1_minus)
        return (total2 - total1) % mod