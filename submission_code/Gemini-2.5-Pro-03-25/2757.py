import sys
# Setting a higher recursion depth might be necessary for extremely deep DP calls, 
# but Python's default is usually sufficient for typical constraints like length 23.
# Uncomment the following line if RecursionError occurs, though it's unlikely for this problem.
# sys.setrecursionlimit(3000) 

class Solution:
    def __init__(self):
        """
        Initializes the Solution object.
        Memoization table 'memo' stores results of computed DP states.
        'S' holds the current number string being processed.
        'MOD' is the modulus value specified in the problem.
        """
        self.memo = {}
        self.S = ""
        self.MOD = 10**9 + 7

    def _solve(self, idx, tight, current_sum, is_leading_zero, min_sum, max_sum):
        """
        The core recursive function implementing the digit DP logic.
        Counts numbers with digit sum constraints up to the number represented by self.S.

        Parameters:
            idx: Current digit index being processed (from left, 0-based).
            tight: Boolean flag. True if the number being built is currently restricted by the digits of self.S.
                   False if we can place any digit (0-9) freely because the prefix is already smaller.
            current_sum: The sum of digits placed so far.
            is_leading_zero: Boolean flag. True if we are currently placing leading zeros (number effectively 0 so far).
            min_sum: Minimum allowed digit sum for a number to be 'good'.
            max_sum: Maximum allowed digit sum for a number to be 'good'.

        Returns:
            The count of valid numbers that can be formed by completing the suffix starting at idx, modulo MOD.
        """
        
        # Pruning optimization: If current_sum exceeds max_sum, no extension can possibly satisfy the constraints.
        if current_sum > max_sum:
            return 0
        
        # Base case: All digits have been processed.
        if idx == len(self.S):
            # Check if a valid number was formed:
            # 1. Must not be composed entirely of leading zeros (unless the number is genuinely 0, handled outside).
            # 2. The total digit sum must be within the [min_sum, max_sum] range.
            if not is_leading_zero and min_sum <= current_sum <= max_sum:
                return 1 # Found one valid number ending this path.
            else:
                return 0 # This path did not result in a valid number.

        # Use a tuple for the state representation to make it hashable for memoization dictionary key.
        state = (idx, tight, current_sum, is_leading_zero)
        # Check if this state has been computed before. If so, return the stored result.
        if state in self.memo:
            return self.memo[state]

        # Determine the upper limit for the digit choice at the current position 'idx'.
        # If 'tight' is True, the limit is the digit S[idx]. Otherwise, the limit is 9.
        limit = int(self.S[idx]) if tight else 9
        
        ans = 0
        # Iterate through all possible digits (0 to 'limit') for the current position.
        for digit in range(limit + 1):
            # Determine the 'tight' constraint for the next recursive call.
            # It remains 'tight' only if the current state was 'tight' AND we chose the maximum possible digit ('limit').
            new_tight = tight and (digit == limit)
            
            # Handle the leading zero state transition.
            if is_leading_zero and digit == 0:
                # If we are placing a leading zero, stay in the leading zero state. Sum remains 0.
                ans = (ans + self._solve(idx + 1, new_tight, current_sum, True, min_sum, max_sum)) % self.MOD
            else:
                # If we place a non-zero digit, or if we were already past leading zeros:
                # - The next state is no longer 'is_leading_zero'.
                # - Update the sum by adding the current 'digit'.
                new_sum = current_sum + digit
                # Make the recursive call for the next digit position with the updated state.
                ans = (ans + self._solve(idx + 1, new_tight, new_sum, False, min_sum, max_sum)) % self.MOD
        
        # Store the computed result for the current state in the memoization table.
        self.memo[state] = ans
        return ans

    def _count_le(self, N_str, min_sum, max_sum):
        """
        Helper function to count numbers x such that 1 <= x <= N_str 
        and min_sum <= digit_sum(x) <= max_sum. It sets up the DP state and calls _solve.
        
        Parameters:
            N_str: The string representation of the upper bound N.
            min_sum: Minimum allowed digit sum.
            max_sum: Maximum allowed digit sum.

        Returns:
            The count of 'good' numbers up to N (inclusive).
        """
        
        # Edge case check: If N_str represents 0. There are no integers x such that 1 <= x <= 0.
        # This check is important for the main calculation logic: count(R) - count(L-1).
        # When L=1, L-1=0. count(0) should return 0.
        # `int(N_str)` handles potentially large number strings correctly.
        if not N_str or int(N_str) == 0: 
             return 0
        
        # Set the global string S for the DP process.
        self.S = N_str
        # Clear the memoization table for a fresh calculation.
        self.memo = {} 
        # Initiate the DP calculation starting from the most significant digit (index 0).
        # Initial state: tight=True (bound by N_str), current_sum=0, is_leading_zero=True.
        return self._solve(0, True, 0, True, min_sum, max_sum)


    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        """
        Main method to calculate the count of 'good' integers x in the range [num1, num2].
        A number x is 'good' if num1 <= x <= num2 and min_sum <= digit_sum(x) <= max_sum.
        The result is returned modulo 10^9 + 7.

        Parameters:
            num1: String representation of the lower bound of the range.
            num2: String representation of the upper bound of the range.
            min_sum: Minimum allowed digit sum.
            max_sum: Maximum allowed digit sum.

        Returns:
            The count of good integers in the specified range, modulo 10^9 + 7.
        """
        
        # The count of good integers in the range [num1, num2] is calculated using the inclusion-exclusion principle:
        # Count(num1, num2) = Count(<= num2) - Count(<= num1 - 1).
        
        # Calculate num1 - 1. Python's int() supports arbitrary precision integers.
        num1_int = int(num1)
        num1_minus_1_int = num1_int - 1
        
        # Convert num1 - 1 back to its string representation.
        # str() correctly handles 0 and positive large integers.
        num1_minus_1_str = str(num1_minus_1_int)
        # If num1 was "1", num1_minus_1_int is 0, and num1_minus_1_str becomes "0".
        
        # Calculate the count of good numbers x such that 1 <= x <= num2.
        ans2 = self._count_le(num2, min_sum, max_sum)
        
        # Calculate the count of good numbers x such that 1 <= x <= num1 - 1.
        # The helper function _count_le handles the case num1_minus_1_str = "0" correctly, returning 0.
        ans1 = self._count_le(num1_minus_1_str, min_sum, max_sum)

        # Compute the final result: (ans2 - ans1).
        # Use modulo arithmetic to handle potential negative results from subtraction
        # and ensure the final answer is non-negative and less than MOD.
        result = (ans2 - ans1 + self.MOD) % self.MOD
        
        return result