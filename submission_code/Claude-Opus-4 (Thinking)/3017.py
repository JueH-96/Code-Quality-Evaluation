class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count_beautiful(n):
            if n == 0:
                return 0
                
            digits = []
            temp = n
            while temp > 0:
                digits.append(temp % 10)
                temp //= 10
            digits.reverse()
            
            from functools import lru_cache
            
            @lru_cache(maxsize=None)
            def dp(pos, started, diff, remainder, tight):
                # pos: current position
                # started: whether we've placed a non-zero digit
                # diff: count of even digits - count of odd digits
                # remainder: current remainder when divided by k
                # tight: whether we're still bounded by the limit
                
                if pos == len(digits):
                    # Check if the conditions are met
                    if started and diff == 0 and remainder == 0:
                        return 1
                    return 0
                
                limit = digits[pos] if tight else 9
                result = 0
                
                for digit in range(0, limit + 1):
                    new_started = started or (digit != 0)
                    
                    # Update the difference
                    if new_started:
                        if digit % 2 == 0:
                            new_diff = diff + 1
                        else:
                            new_diff = diff - 1
                    else:
                        # Leading zero, don't change diff
                        new_diff = diff
                    
                    # Update the remainder
                    new_remainder = (remainder * 10 + digit) % k
                    
                    # Update tight
                    new_tight = tight and (digit == limit)
                    
                    result += dp(pos + 1, new_started, new_diff, new_remainder, new_tight)
                
                return result
            
            return dp(0, False, 0, 0, True)
        
        return count_beautiful(high) - count_beautiful(low - 1)