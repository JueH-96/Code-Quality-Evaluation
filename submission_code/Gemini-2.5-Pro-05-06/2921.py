class Solution:
    MOD = 10**9 + 7

    def _solve(self, s_str: str) -> int:
        # If s_str is "0", count of positive stepping numbers up to 0 is 0.
        if s_str == "0":
            return 0
        
        s_digits = [int(d) for d in s_str]
        N = len(s_digits)
        memo = {} # Memoization dictionary for _dfs states

        # _dfs computes the count of stepping numbers.
        # Parameters:
        #   idx: Current digit index being placed (from left, 0 to N-1).
        #   prev_digit: The digit placed at index idx-1.
        #               A dummy value (10) is used if no actual digit precedes
        #               (i.e., if is_started is False).
        #   is_less: Boolean. True if prefix formed so far is strictly smaller
        #            than s_str's prefix. If True, current digit can be 0-9.
        #            Else, current digit <= s_digits[idx].
        #   is_started: Boolean. True if a non-zero digit has been placed.
        #               Stepping condition (abs diff = 1) applies only if is_started is True.
        def _dfs(idx: int, prev_digit: int, is_less: bool, is_started: bool) -> int:
            state = (idx, prev_digit, is_less, is_started)
            
            if idx == N: # All N digits placed.
                return 1 if is_started else 0 # Count if non-zero number formed.

            if state in memo:
                return memo[state]

            ans = 0
            # Determine upper limit for the current digit.
            upper_bound = s_digits[idx] if not is_less else 9
            
            for digit in range(upper_bound + 1):
                # new_is_less for next recursive call.
                current_is_less = is_less or (digit < upper_bound)
                
                if not is_started:
                    if digit == 0:
                        # Placing a leading zero. Number hasn't "started". prev_digit remains dummy.
                        ans = (ans + _dfs(idx + 1, 10, current_is_less, False)) % Solution.MOD
                    else:
                        # First non-zero digit. Number "starts". Update prev_digit.
                        ans = (ans + _dfs(idx + 1, digit, current_is_less, True)) % Solution.MOD
                else: # is_started is True. prev_digit is actual previous digit.
                    # Current digit must satisfy stepping number condition.
                    if abs(digit - prev_digit) == 1:
                        ans = (ans + _dfs(idx + 1, digit, current_is_less, True)) % Solution.MOD
            
            memo[state] = ans
            return ans

        # Initial call to _dfs. This counts positive stepping numbers X <= int(s_str).
        return _dfs(0, 10, False, False)

    def countSteppingNumbers(self, low: str, high: str) -> int:
        # Calculate count of stepping numbers <= high.
        ans_high = self._solve(high)
        
        # Calculate count of stepping numbers <= low-1.
        low_val = int(low)
        low_minus_1_val = low_val - 1
        s_low_minus_1 = str(low_minus_1_val) # Becomes "0" if low_val was 1.
        
        ans_low_minus_1 = self._solve(s_low_minus_1)
        
        # Result is (count <= high) - (count <= low-1).
        # Add MOD before taking modulo for non-negative result.
        final_ans = (ans_high - ans_low_minus_1 + Solution.MOD) % Solution.MOD
        return final_ans