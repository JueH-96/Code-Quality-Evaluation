class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        
        MOD = 10**9 + 7

        # This is a helper function that counts stepping numbers in the range [0, s]
        # using digit dynamic programming with memoization.
        def solve(s: str) -> int:
            n = len(s)
            memo = {}

            # The state of our DP is (index, prev_digit, is_tight, is_leading_zero)
            # - index: current digit position we are considering (from left)
            # - prev_digit: the digit at the previous position (index-1)
            # - is_tight: boolean, True if we are restricted by the digits of s
            # - is_leading_zero: boolean, True if we can still place leading zeros
            def dp(index: int, prev_digit: int, is_tight: bool, is_leading_zero: bool) -> int:
                if index == n:
                    # We have successfully constructed a valid number.
                    return 1

                state = (index, prev_digit, is_tight, is_leading_zero)
                if state in memo:
                    return memo[state]
                
                res = 0
                # The upper bound for the current digit. If is_tight, it's s[index], else 9.
                limit = int(s[index]) if is_tight else 9
                
                for digit in range(limit + 1):
                    # If we pick a digit smaller than the limit, the subsequent digits
                    # are no longer tightly constrained.
                    new_tight = is_tight and (digit == limit)
                    
                    if is_leading_zero:
                        if digit == 0:
                            # Still placing leading zeros. The number of digits is effectively smaller.
                            # prev_digit remains a placeholder (10) as there's no actual preceding digit.
                            res = (res + dp(index + 1, 10, new_tight, True)) % MOD
                        else:
                            # This is the first non-zero digit of the number.
                            res = (res + dp(index + 1, digit, new_tight, False)) % MOD
                    else:
                        # We are in the middle of a number. Check the stepping condition.
                        if abs(digit - prev_digit) == 1:
                            res = (res + dp(index + 1, digit, new_tight, False)) % MOD
                
                memo[state] = res
                return res

            # Initial call to start the DP. prev_digit=10 is a placeholder.
            return dp(0, 10, True, True)

        # The number of stepping numbers in [low, high] is
        # (count of stepping numbers <= high) - (count of stepping numbers <= low - 1).
        
        # Calculate count of stepping numbers up to `high`. This includes 0.
        count_high = solve(high)
        
        # Calculate count of stepping numbers up to `low - 1`. This also includes 0.
        # Python's int() and str() handle arbitrarily large numbers.
        low_minus_one = str(int(low) - 1)
        count_low_minus_one = solve(low_minus_one)
        
        # The logic for the final answer is:
        # Let g(S) be the count of stepping numbers in [0, S]. Our `solve` function does this.
        # The count of *positive* stepping numbers in [1, S] is g(S) - 1 (for the number 0).
        # Total positive stepping numbers in [low, high] is:
        # (count of positive stepping numbers <= high) - (count of positive stepping numbers <= low-1)
        # = (g(high) - 1) - (g(low-1) - 1)
        # = g(high) - g(low-1)
        # This simplifies the calculation and correctly handles the inclusion of 0 by our DP.
        
        ans = (count_high - count_low_minus_one + MOD) % MOD
        
        return ans