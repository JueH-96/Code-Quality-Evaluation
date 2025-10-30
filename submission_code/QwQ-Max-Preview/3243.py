class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        s_num = int(s)
        m = len(s)
        higher = 10 ** m
        
        # Calculate lower and upper bounds for the prefix
        numerator = start - s_num
        if numerator <= 0:
            lower = 0
        else:
            lower = (numerator + higher - 1) // higher  # Ceiling division
        
        upper_numerator = finish - s_num
        if upper_numerator < 0:
            return 0
        upper = upper_numerator // higher
        
        lower = max(lower, 0)
        upper = min(upper, 10**20 // higher)  # Avoid potential overflow, though Python handles big integers
        
        if lower > upper:
            return 0
        
        # Function to count numbers up to X_str with all digits <= limit and no leading zeros (except zero itself)
        def count_numbers_upto(X_str):
            n = len(X_str)
            from functools import lru_cache
            
            @lru_cache(maxsize=None)
            def dp(pos, tight, leading_zero):
                if pos == n:
                    return 0
                res = 0
                max_d = int(X_str[pos]) if tight else 9
                for d in range(0, max_d + 1):
                    new_tight = tight and (d == max_d)
                    new_leading_zero = leading_zero and (d == 0)
                    if new_leading_zero:
                        res += dp(pos + 1, new_tight, new_leading_zero)
                    else:
                        if leading_zero:
                            # First non-zero digit
                            if d > limit:
                                continue
                            res += dp(pos + 1, new_tight, False)
                        else:
                            if d > limit:
                                continue
                            res += dp(pos + 1, new_tight, False)
                return res
            
            total = dp(0, True, True)
            # Check if zero is allowed (i.e., X >= 0)
            if X_str >= '0':
                total += 1
            return total
        
        # Compute count_upper and count_lower
        if upper == 0:
            # Check if zero is allowed
            count_upper = 1 if 0 <= limit else 0
        else:
            count_upper = count_numbers_upto(str(upper))
        
        if lower == 0:
            count_lower = 0
        else:
            count_lower = count_numbers_upto(str(lower - 1))
        
        return count_upper - count_lower