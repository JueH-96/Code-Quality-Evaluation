class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count_beautiful(limit):
            if limit <= 0:
                return 0
            
            s = str(limit)
            n = len(s)
            memo = {}
            
            def dp(pos, tight, started, diff, mod):
                # Base case: if we've placed all digits
                if pos == n:
                    return 1 if started and diff == 0 and mod == 0 else 0
                
                # Memoization
                if (pos, tight, started, diff, mod) in memo:
                    return memo[(pos, tight, started, diff, mod)]
                
                # Determine the maximum digit we can place
                max_digit = int(s[pos]) if tight else 9
                result = 0
                
                # Try placing each possible digit
                for digit in range(0, max_digit + 1):
                    new_tight = tight and (digit == max_digit)
                    
                    if digit == 0 and not started:
                        # Placing a leading zero
                        result += dp(pos + 1, new_tight, False, diff, mod)
                    else:
                        # Placing a significant digit
                        new_diff = diff + (1 if digit % 2 == 0 else -1)
                        new_mod = (mod * 10 + digit) % k
                        result += dp(pos + 1, new_tight, True, new_diff, new_mod)
                
                memo[(pos, tight, started, diff, mod)] = result
                return result
            
            return dp(0, True, False, 0, 0)
        
        return count_beautiful(high) - count_beautiful(low - 1)