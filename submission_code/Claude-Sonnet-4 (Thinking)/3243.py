class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        import math
        
        k = len(s)
        s_val = int(s)
        power_of_10 = 10 ** k
        
        # Calculate prefix bounds
        min_prefix = max(0, math.ceil((start - s_val) / power_of_10))
        max_prefix = (finish - s_val) // power_of_10
        
        if max_prefix < min_prefix:
            return 0
        
        def count_valid_numbers_up_to(max_val):
            if max_val <= 0:
                return 0
            
            str_max = str(max_val)
            n = len(str_max)
            memo = {}
            
            def dp(pos, tight, started):
                if pos == n:
                    return 1 if started else 0
                
                if (pos, tight, started) in memo:
                    return memo[(pos, tight, started)]
                
                max_digit = int(str_max[pos]) if tight else 9
                result = 0
                
                for digit in range(0, min(max_digit, limit) + 1):
                    new_tight = tight and (digit == max_digit)
                    new_started = started or (digit > 0)
                    result += dp(pos + 1, new_tight, new_started)
                
                memo[(pos, tight, started)] = result
                return result
            
            return dp(0, True, False)
        
        total = 0
        
        # Handle prefix = 0 (number is exactly s)
        if min_prefix == 0 and start <= s_val <= finish:
            total += 1
        
        # Handle prefix >= 1
        min_prefix_for_nonzero = max(1, min_prefix)
        if min_prefix_for_nonzero <= max_prefix:
            total += count_valid_numbers_up_to(max_prefix) - count_valid_numbers_up_to(min_prefix_for_nonzero - 1)
        
        return total