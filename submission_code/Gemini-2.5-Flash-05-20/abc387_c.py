import sys

# Memoization table. Cleared for each top-level get_count_up_to_N call.
memo = {}
# Global string representation of N for the current calculation.
# Used by the recursive solve function.
N_STR = ""
# Global length of N_STR.
N_LEN = 0

def solve(idx, tight, is_leading_zero, top_digit_val, is_snake_broken):
    """
    Recursive function for digit DP.
    Counts numbers from 0 up to N_STR (based on tight constraint)
    that satisfy the snake property (top digit strictly larger than others).

    Args:
        idx (int): Current digit position being considered (0-indexed).
        tight (bool): True if the current prefix matches N_STR's prefix, restricting digit choices.
        is_leading_zero (bool): True if all digits placed so far are zero.
        top_digit_val (int): The value of the most significant digit (first non-zero digit).
                             0 if still in leading zero state.
        is_snake_broken (bool): True if any non-leading zero digit encountered so far
                                is >= top_digit_val.

    Returns:
        int: Count of numbers satisfying the criteria from this state.
    """
    
    # Create a unique state tuple for memoization
    state = (idx, tight, is_leading_zero, top_digit_val, is_snake_broken)
    if state in memo:
        return memo[state]

    # Base case: All digits have been processed
    if idx == N_LEN:
        # A number has been successfully formed.
        # It's a valid candidate if it's not made of only leading zeros
        # (i.e., it's a number like 1, 10, 100, not 0, 00)
        # AND its snake property (top digit > all other digits) is not broken.
        # 1-digit numbers (e.g., 1, 5, 9) will be counted by this function as valid
        # because they satisfy the snake property vacuously. The "not less than 10"
        # constraint is handled in the `get_count_up_to_N` wrapper function.
        return 1 if not is_leading_zero and not is_snake_broken else 0

    ans = 0
    # Determine the upper bound for the current digit.
    # If tight is True, the digit cannot exceed the corresponding digit in N_STR.
    # Otherwise, it can be any digit from 0 to 9.
    upper_bound = int(N_STR[idx]) if tight else 9

    for digit in range(upper_bound + 1):
        # Calculate new state parameters for the recursive call
        new_tight = tight and (digit == upper_bound)
        new_is_leading_zero = is_leading_zero and (digit == 0)
        new_top_digit_val = top_digit_val
        new_is_snake_broken = is_snake_broken

        if is_leading_zero:
            if digit != 0:
                # This is the first non-zero digit encountered, so it becomes the top digit.
                new_top_digit_val = digit
            # If digit is 0, new_top_digit_val remains 0 (placeholder),
            # and new_is_leading_zero remains True.
        else: 
            # We are not in a leading zero state, meaning top_digit_val has been set (> 0).
            # Check if the current digit violates the snake property.
            # A digit violates if it is greater than or equal to the top digit.
            if digit >= top_digit_val:
                new_is_snake_broken = True
        
        # Recursively call for the next digit and add its contribution
        ans += solve(idx + 1, new_tight, new_is_leading_zero, new_top_digit_val, new_is_snake_broken)
    
    # Store and return the result for the current state
    memo[state] = ans
    return ans

def get_count_up_to_N(n_val):
    """
    Calculates the total count of Snake numbers up to n_val (inclusive).

    Args:
        n_val (int): The upper limit (N) for counting Snake numbers.

    Returns:
        int: The count of Snake numbers in the range [10, n_val].
    """
    global N_STR, N_LEN, memo # Declare globals to be modified within this function

    # Snake numbers must be >= 10. If n_val is less than 10, no snake numbers exist.
    if n_val < 10:
        return 0

    N_STR = str(n_val)
    N_LEN = len(N_STR)
    memo = {} # Clear memoization table for each new N_STR to ensure correct calculation

    # Initial call to the recursive DP function:
    # - idx = 0: Start processing from the first digit position.
    # - tight = True: Initially, the number construction is restricted by N_STR.
    # - is_leading_zero = True: Initially, we are in a leading zero state.
    # - top_digit_val = 0: Placeholder for the top digit, will be set on first non-zero digit.
    # - is_snake_broken = False: No digits processed yet, so property is not broken.
    total_valid_numbers = solve(0, True, True, 0, False)

    # The `solve` function counts all numbers X <= N_val that satisfy the
    # "top digit strictly larger than every other digit" property,
    # including 1-digit numbers (1, 2, ..., 9).
    # These 1-digit numbers vacuously satisfy the snake property (no "other digits").
    # However, the problem specifies "A positive integer not less than 10".
    # Therefore, we must exclude these 1-digit numbers. There are 9 such numbers (1 through 9).
    # `max(0, ...)` ensures that if `total_valid_numbers` is <= 9 (e.g., for n_val=5),
    # the result is 0, as no Snake numbers (>=10) would have been found.
    return max(0, total_valid_numbers - 9)

def main():
    # Read L and R from standard input
    L, R = map(int, sys.stdin.readline().split())

    # The count of Snake numbers in [L, R] is (count of Snake numbers up to R) - (count of Snake numbers up to L-1).
    ans = get_count_up_to_N(R) - get_count_up_to_N(L - 1)
    
    # Print the final answer
    print(ans)

if __name__ == '__main__':
    main()