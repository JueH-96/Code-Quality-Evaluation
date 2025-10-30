class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        from functools import lru_cache

        # Helper function to count beautiful numbers up to 'number'
        def count(number):
            s = str(number)
            n = len(s)

            @lru_cache(maxsize=None)
            def dp(pos, diff, mod, tight, started):
                if pos == n:
                    return int(started and diff == 0 and mod == 0)
                
                limit = int(s[pos]) if tight else 9
                total = 0
                for digit in range(0, limit + 1):
                    next_tight = tight and (digit == limit)
                    next_started = started or (digit != 0)
                    
                    if not next_started:
                        next_diff = diff
                    else:
                        if digit % 2 == 0:
                            next_diff = diff + 1
                        else:
                            next_diff = diff - 1
                    
                    next_mod = (mod * 10 + digit) % k
                    total += dp(pos + 1, next_diff, next_mod, next_tight, next_started)
                
                return total

            return dp(0, 0, 0, True, False)
        
        # Handle low=1 to high=10^9
        # Compute count(high) - count(low - 1)
        # Ensure low - 1 is at least 0
        if low == 1:
            lower_count = count(0)
        else:
            lower_count = count(low - 1)
        
        higher_count = count(high)
        
        return higher_count - lower_count