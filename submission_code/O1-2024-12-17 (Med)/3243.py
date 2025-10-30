class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        # Convert s to an integer and determine its length
        s_int = int(s)
        len_s = len(s)
        
        # If s itself is larger than finish, it can never be a suffix of a number in [start..finish]
        if s_int > finish:
            return 0
        
        # Helper function: count how many integers from 0..X (inclusive) have all digits <= limit
        def count_digit_limited(X: int, limit_digit: int) -> int:
            # If X < 0, there are no valid non-negative integers up to a negative number.
            if X < 0:
                return 0
            
            digits = str(X)
            n = len(digits)
            # dp[pos][is_less] = number of ways to form valid numbers from position pos
            # with a flag is_less indicating we've already chosen a digit smaller than the 
            # corresponding digit of X at some earlier position.
            dp = [[-1, -1] for _ in range(n + 1)]
            
            def dfs(pos: int, is_less: bool) -> int:
                if pos == n:
                    return 1
                if dp[pos][is_less] != -1:
                    return dp[pos][is_less]
                
                limit_here = int(digits[pos]) if not is_less else 9
                total_ways = 0
                for dig in range(limit_here + 1):
                    if dig <= limit_digit:
                        total_ways += dfs(pos + 1, is_less or (dig < limit_here))
                
                dp[pos][is_less] = total_ways
                return total_ways
            
            return dfs(0, False)
        
        # We want to find all x = P * 10^len_s + s_int in [start..finish]
        # => start <= P * 10^len_s + s_int <= finish
        # => (start - s_int) / 10^len_s <= P <= (finish - s_int) / 10^len_s
        # P must be an integer >= 0.
        
        power_of_10 = 10 ** len_s
        
        # Compute P_max = floor((finish - s_int) / 10^len_s)
        P_max = (finish - s_int) // power_of_10
        if P_max < 0:
            # No valid prefix if even the smallest prefix yields a number > finish
            return 0
        
        # Compute P_min = ceil((start - s_int) / 10^len_s), clamped to at least 0
        # Using integer arithmetic, ceil(a/b) for positive a,b can be done as (a + b - 1)//b
        # If (start - s_int) <= 0, we just set P_min = 0
        diff = start - s_int
        if diff > 0:
            P_min = (diff + power_of_10 - 1) // power_of_10
        else:
            P_min = 0
        
        if P_min > P_max:
            return 0
        
        # Count how many integers in [P_min..P_max] have all digits <= limit
        # = count_digit_limited(P_max) - count_digit_limited(P_min - 1)
        # handling the lower-bound carefully
        total_up_to_P_max = count_digit_limited(P_max, limit)
        total_up_to_P_min_minus_1 = count_digit_limited(P_min - 1, limit) if P_min > 0 else 0
        
        return total_up_to_P_max - total_up_to_P_min_minus_1