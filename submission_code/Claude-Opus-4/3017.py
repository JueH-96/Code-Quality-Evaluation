class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count_beautiful(n):
            s = str(n)
            
            @cache
            def dp(pos, even_count, odd_count, remainder, tight, started):
                # Base case: reached end of number
                if pos == len(s):
                    # Check if we have a valid number (started) with equal even/odd counts and divisible by k
                    return int(started and even_count == odd_count and remainder == 0)
                
                # Determine the maximum digit we can place at this position
                max_digit = int(s[pos]) if tight else 9
                
                result = 0
                for digit in range(0, max_digit + 1):
                    # Skip leading zeros unless we haven't started the number yet
                    if digit == 0 and not started:
                        result += dp(pos + 1, even_count, odd_count, remainder, tight and digit == max_digit, False)
                    else:
                        # Update counts based on whether digit is even or odd
                        new_even = even_count + (1 if digit % 2 == 0 else 0)
                        new_odd = odd_count + (1 if digit % 2 == 1 else 0)
                        # Update remainder
                        new_remainder = (remainder * 10 + digit) % k
                        # Continue to next position
                        result += dp(pos + 1, new_even, new_odd, new_remainder, tight and digit == max_digit, True)
                
                return result
            
            return dp(0, 0, 0, 0, True, False)
        
        # Count beautiful numbers from 0 to high minus count from 0 to low-1
        return count_beautiful(high) - count_beautiful(low - 1)