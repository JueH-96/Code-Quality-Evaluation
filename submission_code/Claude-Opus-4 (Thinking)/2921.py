class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def count_stepping(num_str, inclusive=True):
            n = len(num_str)
            from functools import lru_cache
            
            @lru_cache(None)
            def dp(pos, last_digit, tight, started):
                if pos == n:
                    if not started:
                        return 0
                    if not inclusive and tight:
                        return 0
                    return 1
                
                limit = int(num_str[pos]) if tight else 9
                res = 0
                
                for digit in range(0, limit + 1):
                    new_tight = tight and (digit == limit)
                    
                    if not started:
                        if digit == 0:
                            res = (res + dp(pos + 1, -1, new_tight, False)) % MOD
                        else:
                            res = (res + dp(pos + 1, digit, new_tight, True)) % MOD
                    else:
                        if abs(last_digit - digit) == 1:
                            res = (res + dp(pos + 1, digit, new_tight, True)) % MOD
                
                return res
            
            return dp(0, -1, True, False)
        
        # Count stepping numbers <= high
        count_high = count_stepping(high, inclusive=True)
        
        # Count stepping numbers < low
        count_below_low = count_stepping(low, inclusive=False)
        
        return (count_high - count_below_low) % MOD