class Solution:
    MOD = 10**9 + 7
    
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        adjusted_min = max(min_sum, 0)
        adjusted_max = min(max_sum, 198)
        
        if adjusted_min > adjusted_max:
            return 0
        
        num1_minus_1 = self.subtract_one(num1)
        
        count_hi = self.helper(num2, adjusted_min, adjusted_max)
        count_lo = self.helper(num1_minus_1, adjusted_min, adjusted_max)
        
        return (count_hi - count_lo) % self.MOD
    
    def subtract_one(self, s: str) -> str:
        s_list = list(s)
        i = len(s_list) - 1
        while i >= 0 and s_list[i] == '0':
            s_list[i] = '9'
            i -= 1
        if i < 0:
            return '0'
        s_list[i] = str(int(s_list[i]) - 1)
        result = ''.join(s_list).lstrip('0')
        return result if result else '0'
    
    def helper(self, x_str: str, adjusted_min: int, adjusted_max: int) -> int:
        digits = list(map(int, x_str))
        n = len(digits)
        
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(pos: int, tight: bool, sum_so_far: int, started: bool) -> int:
            if pos == n:
                return 1 if (adjusted_min <= sum_so_far <= adjusted_max) else 0
            
            res = 0
            max_d = digits[pos] if tight else 9
            
            for d in range(0, max_d + 1):
                new_tight = tight and (d == max_d)
                new_started = started or (d != 0)
                new_sum = sum_so_far + (d if new_started else 0)
                
                if new_sum > adjusted_max:
                    continue
                
                remaining = n - pos - 1
                if new_sum + 9 * remaining < adjusted_min:
                    continue
                
                res += dp(pos + 1, new_tight, new_sum, new_started)
                res %= self.MOD
            
            return res % self.MOD
        
        return dp(0, True, 0, False)