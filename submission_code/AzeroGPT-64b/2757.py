from functools import cache

class Solution:
    def count(self, num1: str, num2: str, lower: int, upper: int) -> int:
        """
        Counts the number of integers between num1 and num2 (inclusive) whose digit sum is within [lower, upper].
        This solution uses a dynamic programming approach with memoization for efficiency.
        """
        MOD = 10 ** 9 + 7
        
        @cache
        def DP(n: int, lower_bound: int, upper_bound: int, allow_lower_exceed: bool, allow_upper_exceed: bool, current_sum: int) -> int:
            if n == 0:
                return int(allow_lower_exceed and allow_upper_exceed and lower <= current_sum <= upper)
            ans = 0
            digit_base = int(upper_bound[n-1]) if allow_upper_exceed else 9
            for d in range(0, digit_base + 1):
                new_lower_exceed = allow_lower_exceed and (current_sum + d >= lower)
                new_upper_exceed = allow_upper_exceed and (current_sum + d <= upper)
                if not allow_lower_exceed and current_sum + d < lower:
                    continue
                if not allow_upper_exceed and current_sum + d > upper:
                    continue
                next_digit = upper_bound[n-1] if allow_upper_exceed and d == digit_base else '9'
                ans += DP(n-1, lower_bound, next_digit, new_lower_exceed, new_upper_exceed, current_sum + d)
            return ans % MOD
        
        return DP(len(num2), num1, num2, True, True, 0)