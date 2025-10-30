# YOUR CODE HERE
import sys

# Increase recursion depth limit for deep recursions. 
# While the maximum depth is related to the number of digits (up to 19 for 10^18),
# setting a higher limit is a safeguard against potential RecursionError on platforms 
# with lower default limits.
try:
    # Set recursion depth to a value slightly larger than the maximum possible depth.
    # 2000 is generally safe for competitive programming contexts.
    sys.setrecursionlimit(2000) 
except Exception as e: 
    # Handle potential exceptions if the platform restricts setting the recursion depth.
    # E.g., some online judges might not allow this. In most cases, this line is safe.
    pass 

# Using a dictionary for memoization. The keys are state tuples, and values are the computed counts.
memo = {}
# Global variable to store the string representation of N (the upper bound for counting).
# This avoids passing the string repeatedly in recursive calls.
S = ""

def solve(idx, tight, is_leading, top_digit_val):
    """
    Recursive DP function to count numbers X <= N (represented by global string S) 
    that satisfy the property: the top digit (first non-zero digit) is strictly 
    greater than all subsequent digits.
    This count includes single-digit positive numbers (1 through 9) if N >= 1, as they technically
    satisfy the property (vacuously true or by interpreting the single digit as the 'top digit').
    
    Args:
        idx: Current digit index being processed (from left, 0-based). Index ranges from 0 to len(S)-1.
        tight: Boolean flag. True if the number constructed so far matches the prefix of N.
               This means the current digit choice is limited by N's digit at this index.
               False if the number constructed is already strictly smaller than N's prefix,
               allowing any digit from 0 to 9 for the current position.
        is_leading: Boolean flag. True if we are currently placing leading zeros (i.e., we haven't 
                    placed any non-zero digit yet).
        top_digit_val: Integer. Stores the value of the first non-zero digit placed. It's -1 if 
                       no non-zero digit has been placed yet (i.e., if `is_leading` is True).

    Returns:
        The count of numbers that can be formed from the current state onwards, satisfying 
        the digit property and the upper bound constraint N.
    """
    
    # Base case: If we have processed all digits (index reaches the length of the number string)
    if idx == len(S):
        # If `is_leading` is True, it means the number formed consists only of zeros (value 0). 
        # We don't count 0 based on problem constraints (N >= 10).
        # If `is_leading` is False, it means a positive number has been formed which satisfies the property.
        # We return 1 to count this number. This count includes single-digit numbers (1-9).
        return 1 if not is_leading else 0

    # Memoization check: If this state has already been computed, return the stored result.
    state = (idx, tight, is_leading, top_digit_val)
    if state in memo:
        return memo[state]

    res = 0
    # Determine the upper limit for the current digit `d`.
    # If `tight` is True, the limit is the digit S[idx] from N. Otherwise, it's 9.
    limit = int(S[idx]) if tight else 9

    # Iterate through all possible digits `d` for the current position `idx` (from 0 up to `limit`).
    for d in range(limit + 1):
        # Calculate the 'tight' constraint for the next recursive call.
        # It remains True only if the current state is tight AND we choose the maximum possible digit `d == limit`.
        new_tight = tight and (d == limit)
        
        if is_leading:
            # If we are currently in the leading zero phase:
            if d == 0:
                # If we place another 0, we stay in the leading zero phase.
                res += solve(idx + 1, new_tight, True, -1)
            else:
                # If we place a non-zero digit `d`, this is the first non-zero digit (top digit).
                # We exit the leading zero phase. The `top_digit_val` is set to `d`.
                # Subsequent digits must be strictly less than `d` to satisfy the property.
                res += solve(idx + 1, new_tight, False, d)
        else:
            # If we are not in the leading zero phase, `top_digit_val` is already set.
            # Check if the current digit `d` satisfies the Snake property: `d` must be strictly less than `top_digit_val`.
            if d < top_digit_val:
                # If the property is satisfied, recurse for the next digit position.
                # The state remains non-leading, and `top_digit_val` doesn't change.
                res += solve(idx + 1, new_tight, False, top_digit_val)
            # else: If `d >= top_digit_val`, this path violates the Snake property. 
            # We add 0 to the result (effectively doing nothing for this choice of `d`).

    # Store the computed result for the current state in the memoization table before returning.
    memo[state] = res
    return res

def count_snake(N):
    """
    Calculates the count of Snake numbers X such that 10 <= X <= N.
    A Snake number is defined as a positive integer >= 10 whose most significant digit 
    (top digit) in decimal representation is strictly larger than every other digit in that number.
    
    Args:
        N: The upper bound integer (inclusive).
        
    Returns:
        The count of Snake numbers in the range [10, N]. Returns 0 if N < 10.
    """
    # According to the problem definition, Snake numbers must be >= 10.
    # If N < 10, there are no Snake numbers in the range [10, N].
    if N < 10:
        return 0
    
    # Prepare for the DP calculation:
    # Set the global string S to the string representation of N.
    # Clear the memoization dictionary for the new calculation.
    global S, memo
    S = str(N)
    memo = {} 
    
    # Call the DP function `solve` starting from the first digit (index 0).
    # Initial state: index 0, tight constraint is True, is_leading is True, top_digit_val is -1.
    # `dp_count` will be the count of numbers X such that 0 <= X <= N AND X satisfies the digit property.
    # This count includes single-digit positive numbers (1 through 9).
    dp_count = solve(0, True, True, -1)
    
    # The problem defines Snake numbers for integers >= 10.
    # Our DP function `solve` counts numbers satisfying the property, including single-digit numbers 1 through 9.
    # These single-digit numbers are not Snake numbers by definition (require N >= 10).
    # We need to subtract the count of these single-digit numbers from the `dp_count`.
    # Since the function is called with N >= 10, the range [0, N] always includes all nine single-digit positive numbers (1..9).
    # Therefore, the DP always counts these 9 numbers.
    single_digit_count = 9 
    
    # The final count of Snake numbers (which are >= 10) up to N is the total DP count minus 9.
    return dp_count - single_digit_count

# Main part of the script that executes when run.
if __name__ == '__main__':
    # Read the input values L and R from standard input.
    L, R = map(int, sys.stdin.readline().split())

    # To find the count of Snake numbers in the range [L, R] (inclusive),
    # we use the principle of inclusion-exclusion:
    # Count in [L, R] = Count up to R - Count up to L-1
    # Calculate count_snake(R) and count_snake(L - 1).
    result = count_snake(R) - count_snake(L - 1)

    # Print the final result to standard output.
    print(result)