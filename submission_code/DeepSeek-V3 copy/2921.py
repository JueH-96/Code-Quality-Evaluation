from functools import lru_cache

class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def count_stepping_numbers(s):
            n = len(s)
            @lru_cache(maxsize=None)
            def dp(index, prev_digit, tight, leading_zero):
                if index == n:
                    return 1 if not leading_zero else 0
                limit = int(s[index]) if tight else 9
                total = 0
                for digit in range(0, limit + 1):
                    new_tight = tight and (digit == limit)
                    new_leading_zero = leading_zero and (digit == 0)
                    if leading_zero:
                        total += dp(index + 1, digit, new_tight, new_leading_zero)
                    else:
                        if abs(digit - prev_digit) == 1:
                            total += dp(index + 1, digit, new_tight, new_leading_zero)
                    total %= MOD
                return total
            return dp(0, -1, True, True)
        
        def is_stepping_number(num):
            s = str(num)
            for i in range(1, len(s)):
                if abs(int(s[i]) - int(s[i-1])) != 1:
                    return False
            return True
        
        low_num = int(low)
        high_num = int(high)
        
        # Calculate the count of stepping numbers up to high
        count_high = count_stepping_numbers(high)
        
        # Calculate the count of stepping numbers up to low - 1
        count_low_minus_1 = count_stepping_numbers(str(low_num - 1)) if low_num > 1 else 0
        
        # The result is the difference
        result = (count_high - count_low_minus_1) % MOD
        
        # Handle the case where low is a stepping number
        if is_stepping_number(low_num):
            result = (result + 1) % MOD
        
        return result