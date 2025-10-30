# YOUR CODE HERE
import sys

def is_repdigit(n):
    """Checks if a positive integer n is a repdigit.
    A repdigit has all its digits the same (e.g., 1, 22, 333).
    Returns (True, digit_char) if n is a repdigit, where digit_char is the repeating digit as a character.
    Returns (False, None) otherwise.
    """
    s = str(n)
    # Since n >= 1 is guaranteed by the problem constraints (month and day numbers),
    # the string s will not be empty.
    # We can check if all characters in the string are the same by converting it to a set.
    # If the set has only one element, all digits were the same.
    if len(set(s)) == 1:
        # Return True and the repeating digit (which is the first character).
        return True, s[0]
    else:
        # If the set has more than one element, the digits were not all the same.
        return False, None

def solve():
    # Read the number of months, N.
    n = int(sys.stdin.readline())
    # Read the list of days for each month (D_1, D_2, ..., D_N).
    # d_list is 0-indexed, so d_list[i] corresponds to month i+1.
    # Example: For N=12, d_list will have 12 elements, d_list[0] for month 1, d_list[11] for month 12.
    d_list = list(map(int, sys.stdin.readline().split()))

    # Initialize the counter for dates that satisfy the repdigit condition.
    repdigit_date_count = 0

    # Iterate through each month number from 1 to N.
    for month_num in range(1, n + 1):
        # Check if the current month number is a repdigit.
        is_rep_month, rep_digit_char = is_repdigit(month_num)

        # If the month number is a repdigit:
        if is_rep_month:
            # The month number consists of repeating digits `rep_digit_char`.
            # We need to find days `j` for this month such that `j` also consists
            # only of the digit `rep_digit_char`, and `1 <= j <= D[month_num]`.

            # Get the number of days in the current month (month_num).
            # The index into the 0-indexed d_list is month_num - 1.
            days_in_month = d_list[month_num - 1]

            # Consider potential repdigit days using the same digit `rep_digit_char`.
            # Since the maximum number of days in a month (D_i) is 100,
            # we only need to consider 1-digit and 2-digit repdigit days.

            # Potential Day 1: The single digit value (e.g., 1, 2, ..., 9).
            # This corresponds to the repeating digit itself.
            potential_day1 = int(rep_digit_char)

            # Check if this single-digit day is valid for the current month.
            # The condition is 1 <= potential_day1 <= days_in_month.
            # Since rep_digit_char is from '1' to '9' (because month_num >= 1),
            # potential_day1 is always >= 1. We only need to check the upper bound.
            if potential_day1 <= days_in_month:
                repdigit_date_count += 1

            # Potential Day 2: The double digit value (e.g., 11, 22, ..., 99).
            # This is formed by repeating the digit character twice.
            potential_day2 = int(rep_digit_char + rep_digit_char)

            # Check if this double-digit day is valid for the current month.
            # The condition is 1 <= potential_day2 <= days_in_month.
            # potential_day2 is always >= 11, so the lower bound check (>= 1) is implicitly true.
            # We only need to check the upper bound.
            if potential_day2 <= days_in_month:
                # Add to count if valid. There's no risk of double counting
                # potential_day1 and potential_day2 because potential_day1 (1-9)
                # can never be equal to potential_day2 (11-99).
                repdigit_date_count += 1

    # Print the final total count of repdigit dates.
    print(repdigit_date_count)

# Execute the solve function to run the program.
solve()

# END OF YOUR CODE HERE