class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        # Convert s to an integer
        s_val = int(s)
        
        # If s itself is already larger than finish, there can be no number in [start..finish] ending with s
        if s_val > finish:
            return 0

        # Power of 10 corresponding to the length of s
        p10 = 10 ** len(s)

        # Helper function to compute the integer ceiling of (a / b) using only integer arithmetic.
        # This returns the smallest integer >= (a / b), for b > 0.
        def ceil_div(a, b):
            d, r = divmod(a, b)
            return d if r == 0 else d + 1

        # Digit-DP function to count how many integers from 0..n (inclusive) have all digits <= limit.
        def count_valid(n: int, limit_digit: int) -> int:
            if n < 0:
                return 0
            
            digits = str(n)
            
            from functools import lru_cache
            @lru_cache(None)
            def dfs(pos, is_tight):
                if pos == len(digits):
                    return 1
                # If we are still tight, we can only place up to digits[pos], otherwise up to 9
                max_digit = int(digits[pos]) if is_tight else 9
                total = 0
                for d in range(max_digit + 1):
                    # We cannot use digits above limit_digit
                    if d > limit_digit:
                        break
                    total += dfs(pos + 1, is_tight and (d == max_digit))
                return total

            return dfs(0, True)

        # Compute adjusted bounds for p where x = p * 10^(len(s)) + s_val
        # so that start <= x <= finish  =>  start - s_val <= p*10^len(s) <= finish - s_val
        # => (start - s_val) / 10^len(s) <= p <= (finish - s_val) / 10^len(s)
        L_f = start - s_val
        R_f = finish - s_val
        
        # Floor for the upper bound
        R = R_f // p10
        if R < 0:
            return 0  # No valid p if upper bound is negative

        # Ceiling for the lower bound
        L = ceil_div(L_f, p10)
        
        # p cannot be negative, so clamp L at 0
        L = max(L, 0)
        if L > R:
            return 0

        # Count how many p in [L..R] have all digits <= limit
        # = count_valid(R) - count_valid(L-1)
        return count_valid(R, limit) - count_valid(L - 1, limit)