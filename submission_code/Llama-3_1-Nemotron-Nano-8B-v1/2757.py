class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        n1 = int(num1)
        n2 = int(num2)
        num1_minus_1 = n1 - 1
        
        a = self.compute_f(n2, min_sum, max_sum)
        b = self.compute_f(num1_minus_1, min_sum, max_sum)
        
        return (a - b) % MOD
    
    def compute_f(self, N: int, min_sum: int, max_sum: int) -> int:
        if N < 0:
            return 0
        digits = list(map(int, str(N)))
        
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(pos, tight, sum_so_far, leading_zero):
            if pos == len(digits):
                if leading_zero:
                    return 1 if (min_sum <= 0 <= max_sum) else 0
                else:
                    return 1 if (min_sum <= sum_so_far <= max_sum) else 0
            
            res = 0
            upper = digits[pos] if tight else 9
            
            for d in range(0, upper + 1):
                new_tight = tight and (d == upper)
                new_leading_zero = leading_zero and (d == 0)
                new_sum = sum_so_far
                
                if not new_leading_zero:
                    new_sum += d
                
                if new_sum > max_sum:
                    continue
                
                remaining = len(digits) - pos - 1
                max_possible = new_sum + 9 * remaining
                if max_possible < min_sum:
                    continue
                
                res += dp(pos + 1, new_tight, new_sum, new_leading_zero)
                res %= MOD
            
            return res % MOD
        
        return dp(0, True, 0, True)