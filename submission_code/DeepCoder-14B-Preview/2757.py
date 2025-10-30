class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def subtract_one(s):
            digits = list(map(int, s))
            i = len(digits) - 1
            while i >= 0:
                if digits[i] == 0:
                    digits[i] = 9
                    i -= 1
                else:
                    digits[i] -= 1
                    break
            res = ''.join(map(str, digits)).lstrip('0')
            if not res:
                return '0'
            return res
        
        def helper(num_str):
            digits = list(map(int, num_str))
            n = len(digits)
            
            from functools import lru_cache
            
            @lru_cache(maxsize=None)
            def dp(pos, tight, sum_so_far, started):
                if pos == n:
                    if not started:
                        current_sum = 0
                    else:
                        current_sum = sum_so_far
                    if min_sum <= current_sum <= max_sum:
                        return 1
                    else:
                        return 0
                res = 0
                max_d = digits[pos] if tight else 9
                for d in range(0, max_d + 1):
                    new_tight = tight and (d == max_d)
                    new_started = started or (d != 0)
                    
                    if new_started:
                        if started:
                            new_sum = sum_so_far + d
                        else:
                            new_sum = d
                    else:
                        new_sum = 0
                    
                    if new_sum > max_sum:
                        continue
                    remaining = n - pos - 1
                    if new_sum + remaining * 9 < min_sum:
                        continue
                    
                    res += dp(pos + 1, new_tight, new_sum, new_started)
                return res
            
            return dp(0, True, 0, False)
        
        num1_minus_1 = subtract_one(num1)
        a = helper(num2)
        b = helper(num1_minus_1) if num1 != '0' else 0
        return (a - b) % MOD