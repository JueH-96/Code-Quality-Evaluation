class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        num_s = int(s)
        L = len(s)
        base = 10 ** L
        
        if finish < num_s:
            return 0
            
        if start <= num_s:
            p_min = 0
        else:
            p_min = (start - num_s + base - 1) // base
        
        p_max = (finish - num_s) // base
        
        if p_min > p_max:
            return 0
        
        def F(n):
            if n < 0:
                return 0
            s_n = str(n)
            digits = list(map(int, s_n))
            m = len(digits)
            from functools import lru_cache
            @lru_cache(maxsize=None)
            def dp(i, tight):
                if i == m:
                    return 1
                res = 0
                upper_bound = digits[i] if tight else 9
                upper_bound = min(upper_bound, limit)
                for d in range(0, upper_bound + 1):
                    new_tight = tight and (d == digits[i])
                    res += dp(i + 1, new_tight)
                return res
            return dp(0, True)
        
        total = 0
        if p_min <= 0 <= p_max:
            total += 1
        
        A = max(1, p_min)
        B = p_max
        if A <= B:
            total += F(B) - F(A - 1)
        
        return total