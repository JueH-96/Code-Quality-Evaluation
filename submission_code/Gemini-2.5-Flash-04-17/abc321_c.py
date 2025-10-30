# YOUR CODE HERE
import sys

# List to store generated 321-like numbers
three_two_one_like_numbers = []

def generate_321_like(current_num, last_digit):
    """
    Recursively generates 321-like numbers by appending digits.
    A number is 321-like if its digits are strictly decreasing.
    This function builds numbers digit by digit from left to right.

    Args:
        current_num: The number constructed so far (integer).
                     Initially 0 for the empty prefix.
        last_digit: The last digit appended to current_num (integer).
                    Used to ensure the next digit is strictly smaller.
                    Initially 10 (or any value > 9) for the empty prefix,
                    as the first digit can be any from 1 to 9.
    """
    # Add the current number to the list if it's a positive number.
    # The initial call is generate(0, 10), 0 is not added.
    # All numbers formed by appending digits to a non-zero prefix are positive.
    if current_num > 0:
        three_two_one_like_numbers.append(current_num)

    # Determine the possible next digits.
    # A digit can be appended only if it's strictly less than the last digit used.
    # The possible digits are from 0 up to last_digit - 1.
    # The range function `range(last_digit)` provides numbers from 0 to last_digit - 1.
    for next_digit in range(last_digit):
        # The first digit of a positive number cannot be 0.
        # If current_num is 0, we are choosing the very first digit.
        # If next_digit is 0 in this case, we skip it.
        if current_num == 0 and next_digit == 0:
            continue

        # Construct the new number by appending the next digit.
        new_num = current_num * 10 + next_digit

        # Recursively call the function with the new number and the appended digit.
        # The newly appended digit `next_digit` becomes the `last_digit` for the next level of recursion.
        generate_321_like(new_num, next_digit)

# Start the generation process.
# We start with an empty number (0) and a 'last_digit' of 10, which allows any digit from 0-9
# to be considered as the first digit in the loop `range(last_digit)`.
# The check `if current_num == 0 and next_digit == 0:` prevents a leading zero.
generate_321_like(0, 10)

# The list `three_two_one_like_numbers` now contains all positive 321-like numbers,
# but not necessarily in sorted order.
# Sort the list numerically.
three_two_one_like_numbers.sort()

# Read the input K.
K = int(sys.stdin.readline())

# The K-th smallest number is at index K-1 in the sorted list.
print(three_two_one_like_numbers[K - 1])