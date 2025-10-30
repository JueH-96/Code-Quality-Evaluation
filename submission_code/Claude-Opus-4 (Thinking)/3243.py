class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        s_val = int(s)
        m = len(s)
        
        # Check if s itself has all digits <= limit
        for digit in s:
            if int(digit) > limit:
                return 0
        
        # If finish < s_val, no number in [start, finish] can end with s
        if finish < s_val:
            return 0
        
        # For a number to end with s, it must be of the form k * 10^m + s_val
        # We need start <= k * 10^m + s_val <= finish
        # So: (start - s_val) / 10^m <= k <= (finish - s_val) / 10^m
        
        k_min = 0 if start <= s_val else (start - s_val + 10**m - 1) // 10**m
        k_max = (finish - s_val) // 10**m
        
        if k_max < k_min:
            return 0
        
        # Count k in [k_min, k_max] where all digits of k are <= limit
        count_max = self.count_with_digit_limit(k_max, limit)
        count_min = 0 if k_min == 0 else self.count_with_digit_limit(k_min - 1, limit)
        return count_max - count_min
    
    def count_with_digit_limit(self, n, limit):
        # Count numbers from 0 to n where all digits are <= limit
        if n < 0:
            return 0
        
        s = str(n)
        memo = {}
        
        def dp(pos, is_tight, has_started):
            if pos == len(s):
                # Count the number if it has started (not all leading zeros)
                return 1 if has_started else 0
            
            if (pos, is_tight, has_started) in memo:
                return memo[(pos, is_tight, has_started)]
            
            max_digit = int(s[pos]) if is_tight else 9
            max_digit = min(max_digit, limit)
            
            result = 0
            
            # Option 1: Continue with leading zero (if not started)
            if not has_started:
                result += dp(pos + 1, False, False)
            
            # Option 2: Place a digit
            start_digit = 0 if has_started else 1
            for digit in range(start_digit, max_digit + 1):
                new_is_tight = is_tight and (digit == int(s[pos]))
                result += dp(pos + 1, new_is_tight, True)
            
            memo[(pos, is_tight, has_started)] = result
            return result
        
        # Count numbers from 1 to n, then add 1 for 0
        return dp(0, True, False) + 1