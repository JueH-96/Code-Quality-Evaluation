class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count_beautiful(limit):
            if limit < 0:
                return 0
            
            s = str(limit)
            n = len(s)
            
            # Memoization: (pos, tight, diff, remainder, started)
            # diff = count_even - count_odd (can range from -n to n)
            # remainder = current number % k
            # started = whether we've placed a non-zero digit
            memo = {}
            
            def dp(pos, tight, diff, remainder, started):
                if pos == n:
                    # Valid if: started placing digits, equal even/odd counts, divisible by k
                    return 1 if started and diff == 0 and remainder == 0 else 0
                
                if (pos, tight, diff, remainder, started) in memo:
                    return memo[(pos, tight, diff, remainder, started)]
                
                max_digit = int(s[pos]) if tight else 9
                result = 0
                
                for digit in range(0, max_digit + 1):
                    new_tight = tight and (digit == max_digit)
                    new_started = started or (digit > 0)
                    
                    if new_started:
                        # Count this digit
                        new_diff = diff + (1 if digit % 2 == 0 else -1)
                        new_remainder = (remainder * 10 + digit) % k
                    else:
                        # Leading zero, don't count
                        new_diff = diff
                        new_remainder = remainder
                    
                    result += dp(pos + 1, new_tight, new_diff, new_remainder, new_started)
                
                memo[(pos, tight, diff, remainder, started)] = result
                return result
            
            return dp(0, True, 0, 0, False)
        
        return count_beautiful(high) - count_beautiful(low - 1)