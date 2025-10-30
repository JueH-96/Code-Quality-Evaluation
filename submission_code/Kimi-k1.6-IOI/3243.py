class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        s_num = int(s)
        m = len(s)
        power = 10 ** m
        
        # Check if the smallest number ending with s is already larger than finish
        if s_num > finish:
            return 0
        
        # Compute k_min and k_max
        k_min = max(0, (start - s_num + power - 1) // power)
        k_max = (finish - s_num) // power
        
        # If the range for k is invalid, return 0
        if k_max < 0 or k_min > k_max:
            return 0
        
        # Function to count numbers from 0 to n with all digits <= limit
        def count_numbers(n, limit):
            if n < 0:
                return 0
            digits = list(map(int, str(n)))
            length = len(digits)
            # Initialize memo[pos][tight][started]
            memo = [[[-1 for _ in range(2)] for __ in range(2)] for ___ in range(length + 1)]
            
            def dp(pos, tight, started):
                if pos == length:
                    return 1
                if memo[pos][tight][started] != -1:
                    return memo[pos][tight][started]
                max_d = digits[pos] if tight else 9
                total = 0
                for d in range(0, max_d + 1):
                    if d > limit:
                        continue
                    new_tight = 1 if (tight and d == max_d) else 0
                    new_started = 1 if (started or d != 0) else 0
                    total += dp(pos + 1, new_tight, new_started)
                memo[pos][tight][started] = total
                return total
            
            return dp(0, 1, 0)
        
        # Calculate the count of valid k's in [k_min, k_max]
        count_max = count_numbers(k_max, limit)
        count_min_minus_1 = count_numbers(k_min - 1, limit)
        
        return count_max - count_min_minus_1