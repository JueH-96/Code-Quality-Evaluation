import sys

# It's good practice to increase recursion limit if N can be up to 22,
# though Python's default (often 1000 or 3000) might be enough.
# sys.setrecursionlimit(3000) # Example, adjust if needed

class Solution:
    MOD = 10**9 + 7
    
    # These will be instance variables, set by solve_wrapper for each call.
    # self.s_digits: list of integer digits of the current number string s
    # self.N: length of s_digits
    # self.min_s_constraint: min_sum constraint for digit sum (from problem input)
    # self.max_s_constraint: max_sum constraint for digit sum (from problem input)
    # self.memo: memoization table for dp states

    def dp(self, idx: int, current_sum: int, tight: bool, is_zero: bool) -> int:
        state = (idx, current_sum, tight, is_zero)
        if state in self.memo:
            return self.memo[state]

        # Pruning: if current_sum already exceeds max_s_constraint.
        if current_sum > self.max_s_constraint:
            return 0 # No need to memoize, this state is effectively terminal with 0 ways.
        
        # Base case: reached the end of the number.
        if idx == self.N:
            # Check if current_sum is within the desired range [min_s_constraint, max_s_constraint].
            # If is_zero is True, it means the number formed is 0, so current_sum is 0.
            # Problem constraints: 1 <= min_sum <= max_sum <= 400.
            # So, current_sum=0 (for number 0) will not satisfy min_s_constraint <= 0.
            if self.min_s_constraint <= current_sum <= self.max_s_constraint:
                return 1
            else:
                return 0

        ans = 0
        # Determine the upper limit for the current digit.
        limit = self.s_digits[idx] if tight else 9

        for digit in range(limit + 1):
            new_tight = tight and (digit == limit)
            
            if is_zero and digit == 0:
                # Still placing leading zeros. Number is effectively 0.
                # current_sum for next state is 0. is_zero remains True.
                ans = (ans + self.dp(idx + 1, 0, new_tight, True)) % self.MOD
            else:
                # No longer placing leading zeros OR this is the first non-zero digit.
                # Add current digit to sum. is_zero becomes False.
                ans = (ans + self.dp(idx + 1, current_sum + digit, new_tight, False)) % self.MOD
        
        self.memo[state] = ans
        return ans

    # This function calculates count of x in [0, s_str] satisfying digit sum constraints.
    def solve_wrapper(self, s_str: str, min_sum_param: int, max_sum_param: int) -> int:
        self.s_digits = [int(d) for d in s_str]
        self.N = len(self.s_digits)
        self.min_s_constraint = min_sum_param
        self.max_s_constraint = max_sum_param
        self.memo = {}  # Clear memoization table for this specific call.

        # Initial call to dp:
        # idx=0, current_sum=0 (as sum is 0 initially)
        # tight=True (restricted by s_str's digits at the start)
        # is_zero=True (start by potentially placing leading zeros; number is 0 so far)
        return self.dp(0, 0, True, True)

    def decrement(self, s_str: str) -> str:
        n = len(s_str)
        num_arr = [int(c) for c in s_str]

        i = n - 1
        while i >= 0:
            if num_arr[i] == 0:
                num_arr[i] = 9
                i -= 1
            else:
                num_arr[i] -= 1
                break
        
        # If i < 0, original string was like "1000", which became "0999".
        # This implies the first digit was 1 and all others 0.

        # Handle leading zeros from decrement.
        # Example: "100" -> num_arr is [0,9,9].
        # Example: "1" -> num_arr is [0].
        if num_arr[0] == 0 and n > 1: # Potential leading zero and not a single digit "0"
            first_non_zero_idx = 0
            while first_non_zero_idx < n and num_arr[first_non_zero_idx] == 0:
                first_non_zero_idx += 1
            
            if first_non_zero_idx == n:
                # This means num_arr is all zeros (e.g., [0,0,0]).
                # This should only happen if input s_str was "0...0" (not standard for num1 >= "1")
                # or through a bug. For num1 >= "1", if "100" -> [0,9,9], not all zeros.
                # However, to be robust, if it becomes all zeros, it represents "0".
                return "0" 
            
            return "".join(map(str, num_arr[first_non_zero_idx:]))
        else: 
            # num_arr[0] != 0 (no leading zero to remove)
            # OR n == 1 (single digit, e.g. "0" from "1", or "5" from "6").
            return "".join(map(str, num_arr))


    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        # Calculate count of good numbers x where 0 <= x <= num2.
        ans_num2 = self.solve_wrapper(num2, min_sum, max_sum)
        
        # Calculate num1 - 1.
        num1_minus_1_str = self.decrement(num1)
        
        # Calculate count of good numbers x where 0 <= x <= num1_minus_1_str.
        ans_num1_minus_1 = self.solve_wrapper(num1_minus_1_str, min_sum, max_sum)
        
        # The result is (count <= num2) - (count <= num1 - 1).
        # Add MOD before taking modulo to handle potential negative result from subtraction.
        result = (ans_num2 - ans_num1_minus_1 + self.MOD) % self.MOD
        return result