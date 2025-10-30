import sys

# Use a dictionary for memoization for the DP function counting D-digit numbers
# The memoization state depends on the specific number N (string s) passed to count_D_digits_leq_n.
# We can use a global memo and clear it before each call to count_D_digits_leq_n,
# or pass the memo dictionary explicitly. Using a global cleared memo is simpler.
memo_d_digits = {}

# Helper function to count D-digit numbers X <= s that are Snake numbers.
# X is a D-digit number, so its first digit is 1-9.
# Snake property: First digit > all other D-1 digits.
# This function counts numbers X in [10^(D-1), int(s)] that are Snake numbers.
def count_D_digits_leq_n(s):
    D = len(s)
    global memo_d_digits
    memo_d_digits.clear() # Clear memo for the current call

    # DP state: (index, tight, first_digit_value, is_valid_so_far)
    # index: current digit position we are filling (from 1 to D). Index 0 (first digit) is handled outside.
    # tight: True if restricted by the digits of s at current and subsequent positions.
    # first_digit_value: The value of the digit placed at index 0 (1-9). This is fixed throughout the DP for a given branch.
    # is_valid_so_far: True if all digits placed from index 1 up to index-1 are strictly less than first_digit_value.
    # If this becomes False at any point, it stays False.
    def dp(index, tight, first_digit_value, is_valid_so_far):
        # Base case: Finished placing all D digits from index 1 to D-1.
        if index == D:
            # We have successfully formed a D-digit number starting with `first_digit_value`
            # and filled digits at positions 1 to D-1.
            # The number is a Snake number if the property `first_digit_value > all other digits` holds.
            # `is_valid_so_far` being True means this property holds for digits 1 to D-1.
            return 1 if is_valid_so_far else 0

        # If the property `msd > other digits` has already been violated by previous digits,
        # then no matter what digits we place now, the resulting number won't be a Snake number.
        # We can stop exploring this branch early.
        if not is_valid_so_far:
             return 0 # Optimization

        # Memoization key: (index, tight, first_digit_value, is_valid_so_far)
        state = (index, tight, first_digit_value, is_valid_so_far)
        if state in memo_d_digits:
            return memo_d_digits[state]

        ans = 0
        # Upper limit for the digit at the current position `index`.
        # Limited by s[index] if tight is True, otherwise up to 9.
        upper = int(s[index]) if tight else 9

        # Iterate through possible digits 'd' for the current position `index`.
        # Subsequent digits can be from 0 up to `upper`.
        for d in range(0, upper + 1):
            # Determine the new tight constraint for the next recursive call.
            # It remains tight only if it was tight before AND the current digit placed is the upper limit.
            new_tight = tight and (d == upper)

            # Check if placing digit 'd' maintains the snake property.
            # The current digit `d` must be strictly less than `first_digit_value`.
            # The validity `is_valid_so_far` for the prefix (digits 1 to index-1) is maintained
            # only if it was true before AND the current digit `d` is less than `first_digit_value`.
            new_is_valid_so_far = is_valid_so_far and (d < first_digit_value)

            # Recurse to fill the next digit position (index + 1).
            # Pass the same `first_digit_value` as it's fixed for this D-digit number.
            # Pass the updated `new_tight` and `new_is_valid_so_far`.
            ans += dp(index + 1, new_tight, first_digit_value, new_is_valid_so_far)

        # Store the calculated result for the current state in the memoization table.
        memo_d_digits[state] = ans
        return ans

    # To count D-digit numbers <= s that are Snake:
    # We iterate through all possible values for the first digit (at index 0).
    # The first digit of a D-digit number must be between 1 and 9.
    # It also cannot exceed the first digit of 's' if we want the resulting number to be <= s.
    count_D_digits = 0
    first_digit_limit = int(s[0])
    # The initial tight constraint for the first digit relative to s[0] is always True.
    is_tight_at_index_0 = True

    # Iterate through possible values for the first digit (1 to 9, limited by s[0])
    for first_d in range(1, first_digit_limit + 1):
        # If the chosen first digit `first_d` is equal to the first digit of `s`,
        # the tight constraint continues for the next digit positions (from index 1).
        # Otherwise, the number we are building is strictly less than `s` (at the first digit),
        # so the tight constraint becomes False for all subsequent positions.
        new_tight_initial = is_tight_at_index_0 and (first_d == first_digit_limit)

        # Start the DP recursion from the next digit position (index 1).
        # Pass the chosen `first_d` as `first_digit_value`.
        # The `is_valid_so_far` property is True initially because there are no digits after the first one yet (we start from index 1).
        count_D_digits += dp(1, new_tight_initial, first_d, True)

    # Return the total count of D-digit Snake numbers less than or equal to int(s).
    return count_D_digits


# Main function to count Snake numbers X such that 10 <= X <= n.
def count_snake_up_to(n):
    # Snake numbers are defined as positive integers not less than 10.
    # If the upper limit n is less than 10, there are no Snake numbers <= n.
    if n < 10:
        return 0

    # Convert n to its string representation to work with digits easily.
    s = str(n)
    D = len(s) # Number of digits in n.

    # Step 1: Count Snake numbers with fewer than D digits.
    # These are Snake numbers X where 10 <= X < 10^(D-1).
    # A k-digit Snake number (k >= 2) has a first digit M (1-9) and k-1 subsequent digits in {0, ..., M-1}.
    # The number of such k-digit numbers for a fixed M is M^(k-1).
    # Total k-digit Snake numbers = sum(M^(k-1) for M=1 to 9).
    # We sum this count for each number of digits k from 2 up to D-1.
    count_less_digits = 0
    for k in range(2, D): # Iterate through possible number of digits (2, 3, ..., D-1)
        for m in range(1, 10): # Iterate through possible values for the first digit (1 to 9)
            # Add M^(k-1) to the total count of k-digit Snake numbers.
            count_less_digits += m**(k - 1)

    # Step 2: Count D-digit Snake numbers X such that 10^(D-1) <= X <= n.
    # These are numbers with the same number of digits as n, that are less than or equal to n, and are Snake numbers.
    # We use the helper DP function `count_D_digits_leq_n` for this.
    count_D_digits = count_D_digits_leq_n(s)

    # The total count of Snake numbers X such that 10 <= X <= n is the sum of
    # Snake numbers with fewer than D digits and Snake numbers with exactly D digits <= n.
    return count_less_digits + count_D_digits


# Read input from Standard Input.
# L and R are given on a single line separated by space.
L_str, R_str = sys.stdin.readline().split()
# Convert input strings to integers. L and R can be large (up to 10^18).
L = int(L_str)
R = int(R_str)

# The problem asks for the number of Snake numbers in the range [L, R] inclusive.
# This count is equal to (Count of Snake numbers <= R) - (Count of Snake numbers <= L-1).
# Use the `count_snake_up_to` function to calculate the cumulative counts.
answer = count_snake_up_to(R) - count_snake_up_to(L - 1)

# Print the final answer to Standard Output.
print(answer)