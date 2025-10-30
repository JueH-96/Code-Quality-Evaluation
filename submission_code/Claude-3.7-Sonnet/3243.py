class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        s_int = int(s)
        s_len = len(s)
        
        # Check if s itself is valid (all digits ≤ limit)
        for digit in s:
            if int(digit) > limit:
                return 0
        
        # If finish is smaller than s, no powerful integers exist
        if finish < s_int:
            return 0
        
        # Calculate the range of valid prefixes
        # For a number prefix*10^s_len + s_int to be in range [start, finish]:
        lower_bound = (start - s_int) / (10 ** s_len)
        upper_bound = (finish - s_int) / (10 ** s_len)
        
        lower = max(0, math.ceil(lower_bound))
        upper = math.floor(upper_bound)
        
        # No valid prefixes exist
        if lower > upper:
            return 0
        
        # Use digit DP to count valid prefixes
        from functools import lru_cache
        
        def count_up_to(num):
            if num < 0:
                return 0
            
            num_str = str(num)
            
            @lru_cache(None)
            def dp(idx, tight):
                # Base case: we've processed all digits
                if idx == len(num_str):
                    return 1
                
                # Maximum digit we can use at this position
                max_digit = int(num_str[idx]) if tight else limit
                count = 0
                
                for d in range(min(max_digit + 1, limit + 1)):
                    # Whether we're still constrained by the upper bound
                    new_tight = tight and d == int(num_str[idx])
                    count += dp(idx + 1, new_tight)
                
                return count
            
            return dp(0, True)
        
        # Count numbers in range [lower, upper]
        count_upper = count_up_to(upper)
        count_lower = count_up_to(lower - 1) if lower > 0 else 0
        
        return count_upper - count_lower