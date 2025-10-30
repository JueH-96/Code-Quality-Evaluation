class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def count_powerful(upper_bound):
            # Count powerful integers from 1 to upper_bound
            if upper_bound < int(s):
                return 0
            
            upper_str = str(upper_bound)
            n = len(upper_str)
            suffix_len = len(s)
            
            # If upper_bound has fewer digits than suffix, no valid numbers
            if n < suffix_len:
                return 0
            
            # Memoization for digit DP
            # dp[pos][tight] = count of valid numbers
            from functools import lru_cache
            
            @lru_cache(None)
            def dp(pos, tight):
                # If we've placed all digits
                if pos == n:
                    return 1
                
                # If we're in the suffix part
                if pos >= n - suffix_len:
                    suffix_pos = pos - (n - suffix_len)
                    required_digit = int(s[suffix_pos])
                    
                    # Check if this digit is valid
                    if required_digit > limit:
                        return 0
                    
                    # If tight, check if we can place this digit
                    if tight and required_digit > int(upper_str[pos]):
                        return 0
                    
                    # Place the required digit
                    new_tight = tight and (required_digit == int(upper_str[pos]))
                    return dp(pos + 1, new_tight)
                
                # We're in the prefix part - can choose any valid digit
                max_digit = int(upper_str[pos]) if tight else 9
                max_digit = min(max_digit, limit)
                
                result = 0
                # Start from 0 if not the first position, else from 1
                start_digit = 0 if pos > 0 else 1
                
                for digit in range(start_digit, max_digit + 1):
                    new_tight = tight and (digit == int(upper_str[pos]))
                    result += dp(pos + 1, new_tight)
                
                return result
            
            # Count all valid numbers with same or fewer digits
            total = 0
            
            # Count numbers with fewer digits
            for num_digits in range(suffix_len, n):
                if num_digits == suffix_len:
                    # Only the suffix itself if it's valid
                    if int(s) >= 10**(suffix_len-1) and all(int(d) <= limit for d in s):
                        total += 1
                else:
                    # Numbers with num_digits digits
                    @lru_cache(None)
                    def count_with_digits(pos, num_digits):
                        if pos == num_digits:
                            return 1
                        
                        # If in suffix part
                        if pos >= num_digits - suffix_len:
                            suffix_pos = pos - (num_digits - suffix_len)
                            required_digit = int(s[suffix_pos])
                            if required_digit > limit:
                                return 0
                            return count_with_digits(pos + 1, num_digits)
                        
                        # Prefix part
                        result = 0
                        start_digit = 0 if pos > 0 else 1
                        for digit in range(start_digit, min(9, limit) + 1):
                            result += count_with_digits(pos + 1, num_digits)
                        return result
                    
                    total += count_with_digits(0, num_digits)
            
            # Count numbers with exactly n digits up to upper_bound
            total += dp(0, True)
            
            return total
        
        # Count powerful integers from start to finish
        count_finish = count_powerful(finish)
        count_before_start = count_powerful(start - 1) if start > 1 else 0
        
        return count_finish - count_before_start