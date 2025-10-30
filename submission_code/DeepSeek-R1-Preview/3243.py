class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        m = len(s)
        s_num = int(s)
        if s_num > finish:
            return 0
        
        ten_pow_m = 10 ** m
        x_min = max(0, (start - s_num + ten_pow_m - 1) // ten_pow_m)
        x_max = (finish - s_num) // ten_pow_m
        
        if x_min > x_max:
            return 0
        
        def count_le(n, limit):
            if n < 0:
                return 0
            if n == 0:
                return 1  # Only 0, which is allowed
            
            digits = list(map(int, str(n)))
            length = len(digits)
            
            # Count for x=0
            count = 1
            
            # Count numbers with fewer digits
            for l in range(1, length):
                # Numbers with l digits, first digit 1..limit, rest 0..limit
                count += limit * ( (limit + 1) ** (l - 1) )
            
            # Count numbers with the same number of digits as n
            from functools import lru_cache
            
            @lru_cache(maxsize=None)
            def dp(i, tight):
                if i == length:
                    return 1
                res = 0
                upper = digits[i] if tight else limit
                upper = min(upper, limit)
                start_d = 0 if i > 0 else 1
                for d in range(start_d, upper + 1):
                    new_tight = tight and (d == digits[i])
                    res += dp(i + 1, new_tight)
                return res
            
            same_count = dp(0, True)
            count += same_count
            return count
        
        total = count_le(x_max, limit) - count_le(x_min - 1, limit)
        return total