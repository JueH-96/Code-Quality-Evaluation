class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        # Helper to subtract one from a numeric string.
        def subtractOne(s: str) -> str:
            # Convert string to list of digits for mutability.
            digits = list(s)
            i = len(digits) - 1
            while i >= 0:
                if digits[i] != '0':
                    digits[i] = str(int(digits[i]) - 1)
                    break
                else:
                    digits[i] = '9'
                    i -= 1
            # Remove any leading zeros if present.
            # In case the number was "1" -> becomes "0"
            res = "".join(digits).lstrip('0')
            return res if res != "" else "0"
        
        # Helper function that returns count of numbers from 0 to num (inclusive)
        # whose digit sum is <= limit.
        def countUpTo(num: str, limit: int) -> int:
            n = len(num)
            # The maximum possible digit sum for a number with n digits is 9*n
            effective_limit = min(limit, 9 * n)
            # dp[pos][s][tight]: count ways with positions processed = pos,
            # current digit sum = s, and tight indicating whether we are bound by the prefix.
            from functools import lru_cache
            @lru_cache(maxsize=None)
            def dp(pos: int, current_sum: int, tight: int) -> int:
                # If current_sum already exceeds effective_limit, no valid numbers.
                if current_sum > effective_limit:
                    return 0
                if pos == n:
                    # At the end, if current sum is <= limit then count valid.
                    return 1
                res = 0
                max_digit = int(num[pos]) if tight else 9
                for dig in range(0, max_digit + 1):
                    new_tight = tight and (dig == max_digit)
                    res += dp(pos + 1, current_sum + dig, new_tight)
                    res %= MOD
                return res
            return dp(0, 0, 1)
        
        # Helper to get count of numbers <= a given number that have digit sum in [L, R]
        # using countUpTo: since countUpTo returns count with digit sum <= X, we can do:
        # f(R) - f(L-1).
        def countInRange(num: str, L: int, R: int) -> int:
            # if L <= 0, then countUpTo(num, R) gives count of all numbers with sum <= R.
            low = countUpTo(num, L - 1) if L > 0 else 0
            high = countUpTo(num, R)
            return (high - low) % MOD
        
        # Count for a range [0, x] satisfying condition.
        # We want to count numbers x with digit_sum in [min_sum, max_sum]
        def f(num: str) -> int:
            return countInRange(num, min_sum, max_sum)
        
        # Count for numbers in [num1, num2] = f(num2) - f(num1 - 1)
        num1_minus = subtractOne(num1)
        ans = (f(num2) - f(num1_minus)) % MOD
        return ans