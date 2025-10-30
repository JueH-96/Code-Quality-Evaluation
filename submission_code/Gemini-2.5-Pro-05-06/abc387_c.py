import sys

# Memoization table for the dp function
memo = {}
# Globals to store the string representation of N and its length
# These are set by count_snake_up_to_N for each call.
S_global = ""
L_global = 0

def dp(idx, current_top_d, tight, started):
    """
    Calculates the count of numbers satisfying the Snake property digit criteria.
    idx: current digit index being placed (from left, 0 to L_global-1).
    current_top_d: the most significant digit encountered so far.
                   If started is False, this is a dummy value (e.g., 0).
    tight: boolean, True if we are restricted by the digits of S_global (N).
           False if we can place any digit from 0-9.
    started: boolean, True if a non-zero digit has been placed.
    """
    state = (idx, current_top_d, tight, started)
    if state in memo:
        return memo[state]

    # Base case: all digits have been placed
    if idx == L_global:
        # If started is True, a non-zero number satisfying the property was formed.
        # If started is False, the number formed was 0.
        return 1 if started else 0

    # Determine the upper limit for the current digit
    upper_bound_digit = int(S_global[idx]) if tight else 9
    
    ans = 0
    for digit in range(upper_bound_digit + 1):
        # Update the tight constraint for the next state
        new_tight = tight and (digit == upper_bound_digit)
        
        if not started:
            # We are currently placing leading zeros or the first non-zero digit
            if digit == 0:
                # Still in leading zero phase
                ans += dp(idx + 1, 0, new_tight, False) # current_top_d remains dummy 0
            else:
                # This digit is the first non-zero digit (the actual top digit)
                ans += dp(idx + 1, digit, new_tight, True) # current_top_d becomes this digit
        else:
            # A top digit (current_top_d) has already been established
            # Subsequent digits must be strictly smaller than current_top_d
            if digit < current_top_d:
                ans += dp(idx + 1, current_top_d, new_tight, True)
            # Else (digit >= current_top_d): This path violates the Snake number condition. Add 0.

    memo[state] = ans
    return ans

def count_snake_up_to_N(n_val):
    """
    Counts Snake numbers between 10 and n_val (inclusive).
    """
    if n_val < 10: # Snake numbers are defined to be >= 10
        return 0

    global S_global, L_global, memo
    S_global = str(n_val)
    L_global = len(S_global)
    memo = {} # Clear memoization table for each new N

    # dp_result counts numbers x in [1, n_val] that satisfy the property:
    # "the first non-zero digit is strictly larger than all subsequent digits."
    # This count includes single-digit numbers (1-9).
    # It does not count 0.
    dp_result = dp(0, 0, True, False)
    
    # Single-digit positive numbers (1 through 9) satisfy the digit property vacuously.
    # The dp function includes these in its count.
    # Since Snake numbers must be >= 10, these single-digit numbers must be subtracted.
    # The number of such single-digit positives up to n_val is min(n_val, 9).
    num_single_digit_positive_to_subtract = min(n_val, 9)
    
    return dp_result - num_single_digit_positive_to_subtract

def main():
    L_in, R_in = map(int, sys.stdin.readline().split())
    
    ans_R = count_snake_up_to_N(R_in)
    ans_L_minus_1 = count_snake_up_to_N(L_in - 1)
    
    result = ans_R - ans_L_minus_1
    sys.stdout.write(str(result) + "
")

if __name__ == '__main__':
    main()