class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def count_dp(s):
            real_max = min(max_sum, 198)  # 198 is the maximum possible digit sum (22 digits * 9)
            digits = list(map(int, s))
            n = len(digits)
            
            from functools import lru_cache
            
            @lru_cache(maxsize=None)
            def dp(pos, tight, leading_zero, sum_so_far):
                if pos == n:
                    return 1 if (min_sum <= sum_so_far <= real_max) else 0
                
                res = 0
                upper = digits[pos] if tight else 9
                
                for d in range(0, upper + 1):
                    new_tight = tight and (d == upper)
                    new_leading_zero = leading_zero and (d == 0)
                    
                    if new_leading_zero:
                        new_sum = 0
                    else:
                        if leading_zero:
                            new_sum = d
                        else:
                            new_sum = sum_so_far + d
                    
                    if new_sum > real_max:
                        continue
                    
                    res += dp(pos + 1, new_tight, new_leading_zero, new_sum)
                    res %= MOD
                
                return res % MOD
            
            return dp(0, True, True, 0)
        
        num1_minus_1 = str(int(num1) - 1)
        upper = count_dp(num2)
        lower = count_dp(num1_minus_1)
        
        return (upper - lower) % MOD