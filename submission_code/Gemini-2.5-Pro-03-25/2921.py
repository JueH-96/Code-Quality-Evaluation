import sys

# Setting a higher recursion depth might be necessary for large inputs (L=100).
# Python's default limit is often 1000, which should be sufficient for depth L+1 ~ 101.
# However, setting it explicitly can prevent potential stack overflow errors in some environments.
try:
    # Check current limit and increase if it seems low relative to max possible depth (L+1).
    # Increase if less than 200 to be safe. L can be up to 100. Max depth needed is 101.
    # Set limit slightly higher for buffer.
    current_limit = sys.getrecursionlimit()
    if current_limit < 250: # Check if current limit is potentially too low
         sys.setrecursionlimit(250) # Set higher limit
except Exception as e:
    # Ignore if setting recursion limit fails (e.g., due to restricted environment permissions)
    # You could print a warning here if needed for debugging.
    # print(f"Warning: Could not set recursion depth: {e}") 
    pass 

class Solution:
    MOD = 10**9 + 7
    
    def __init__(self):
        """ Initializes the Solution object. """
        # Memoization cache for DP states. It will store results for states (idx, tight, is_leading, prev_digit).
        self.memo = {}
        # The string representation of the number boundary (N). Will be set by `solve`.
        self.s = ""
        # The length of the number string N. Will be set by `solve`.
        self.L = 0

    def solve_dp(self, idx, tight, is_leading, prev_digit):
        """
        Recursive DP function with memoization to count stepping numbers up to N (represented by self.s).

        Args:
            idx: Current digit index being considered (from left, 0-based).
            tight: Boolean flag indicating if we are restricted by the digits of N. 
                   True means we can only place digits up to self.s[idx], False means 0-9.
            is_leading: Boolean flag indicating if we are currently placing leading zeros.
                        True means the number formed so far is effectively 0.
            prev_digit: The digit placed at the previous position (idx-1). Used to check stepping property.
                        -1 indicates no previous digit (either at the start or during leading zeros).

        Returns:
            The count of valid stepping number suffixes that can be formed from this state, modulo MOD.
        """
        # Base case: If we have processed all digits (reached the end of the number string N)
        if idx == self.L:
            # If is_leading is True, it means the number formed consists of all zeros (value 0).
            # Since we are counting positive stepping numbers (>= 1), return 0.
            # Otherwise, we successfully formed a valid positive number. Return 1.
            return 0 if is_leading else 1 
        
        # Create a tuple representing the current state for memoization look-up.
        state = (idx, tight, is_leading, prev_digit)
        # Return the cached result if this state has already been computed.
        if state in self.memo:
            return self.memo[state]
        
        # Determine the upper bound for the digit we can place at the current index `idx`.
        # If `tight` is True, the maximum digit is self.s[idx]. Otherwise, it's 9.
        upper_bound = int(self.s[idx]) if tight else 9
        ans = 0
        
        # Iterate through all possible digits (0 to upper_bound) for the current position `idx`.
        for digit in range(upper_bound + 1):
            # Calculate the tightness constraint for the next recursive call.
            # The next state is tight only if the current state is tight AND we choose the maximum possible digit.
            new_tight = tight and (digit == upper_bound)
            
            if is_leading:
                # If we are currently in the leading zero phase:
                if digit == 0:
                    # If the current digit is 0, continue in the leading zero phase for the next digit.
                    # Pass prev_digit as -1 marker.
                    ans = (ans + self.solve_dp(idx + 1, new_tight, True, -1)) % self.MOD
                else:
                    # If the current digit is non-zero, this is the first significant digit.
                    # Transition out of the leading zero phase. Record this digit as prev_digit.
                    # No stepping property check needed for the very first non-zero digit.
                    ans = (ans + self.solve_dp(idx + 1, new_tight, False, digit)) % self.MOD
            else:
                # If not in the leading zero phase:
                # We must satisfy the stepping number property: |current digit - previous digit| == 1.
                # prev_digit is guaranteed to be a valid digit (0-9) because is_leading is False.
                if abs(digit - prev_digit) == 1:
                    # If the condition is met, recurse for the next digit state.
                    ans = (ans + self.solve_dp(idx + 1, new_tight, False, digit)) % self.MOD
                # If the condition abs(digit - prev_digit) != 1 is not met, this path doesn't form a stepping number.
                # We add 0 to `ans`, effectively doing nothing for this `digit`.
        
        # Store the computed result for the current state in the memoization cache.
        self.memo[state] = ans
        # Return the computed result.
        return ans

    def solve(self, num_str):
        """
        Helper function to initialize state variables (memo, s, L) and call the DP function.
        Calculates the count of stepping numbers x such that 1 <= x <= N, where N is represented by num_str.
        """
        self.memo = {}  # Reset memoization cache for each new calculation (for a different N).
        self.s = num_str
        self.L = len(num_str)
        # Start the DP calculation from index 0, with tight constraint True, is_leading True, and prev_digit -1.
        # The result is the count of stepping numbers in the range [1, N].
        return self.solve_dp(0, True, True, -1)

    def is_stepping(self, num_str: str) -> bool:
        """ 
        Checks if the number represented by the string `num_str` is a stepping number.
        A number is stepping if the absolute difference between adjacent digits is 1.
        """
        N = len(num_str)
        # Single digit numbers are considered stepping numbers according to examples.
        # Problem constraints state low >= 1, so single digit numbers are positive.
        if N <= 1:
             return True 
        
        # Iterate through adjacent pairs of digits.
        for i in range(N - 1):
            # Check if the absolute difference between adjacent digits is exactly 1.
            if abs(int(num_str[i]) - int(num_str[i+1])) != 1:
                # If any pair doesn't satisfy the condition, it's not a stepping number.
                return False
        # If all adjacent pairs satisfy the condition, it is a stepping number.
        return True

    def countSteppingNumbers(self, low: str, high: str) -> int:
        """
        Calculates the count of stepping numbers in the inclusive range [low, high].
        The result is returned modulo 10^9 + 7.
        
        Uses the principle of inclusion-exclusion: Count(low, high) = Count(<= high) - Count(< low).
        This is equivalent to: Count(<= high) - Count(<= low) + (1 if low is stepping else 0).
        """
        
        # Calculate count of stepping numbers x such that 1 <= x <= high
        count_high = self.solve(high)
        
        # Calculate count of stepping numbers x such that 1 <= x <= low
        count_low = self.solve(low)
        
        # Calculate (Count(<= high) - Count(<= low)) % MOD.
        # Add MOD before taking modulo to handle potential negative results from subtraction.
        ans = (count_high - count_low + self.MOD) % self.MOD
        
        # The subtraction `count_high - count_low` gives the count in range (low, high].
        # We need the count in [low, high]. So, if `low` itself is a stepping number, 
        # it should be included in the count. We add 1 back in that case.
        if self.is_stepping(low):
            ans = (ans + 1) % self.MOD
            
        return ans