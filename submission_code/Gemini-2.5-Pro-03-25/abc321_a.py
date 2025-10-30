# YOUR CODE HERE
import sys

def solve():
    """
    Reads an integer N from standard input, checks if it's a 321-like number,
    and prints "Yes" or "No" to standard output.
    """
    # Read the input integer N as a string
    # Reading as string allows easy access to digits
    n_str = sys.stdin.readline().strip()

    # Assume N is a 321-like number initially
    is_321_like = True

    # Get the number of digits
    num_digits = len(n_str)

    # Check the condition for adjacent digits
    # If the number has only one digit (num_digits == 1),
    # the loop range(num_digits - 1) will be range(0), which is empty.
    # In this case, is_321_like remains True, correctly identifying
    # single-digit numbers as 321-like.
    for i in range(num_digits - 1):
        # Compare the i-th digit (from the left/top) with the (i+1)-th digit.
        # Character comparison is sufficient and correct for digits '0' through '9'
        # as their ASCII/Unicode values are ordered numerically.
        # The condition for 321-like is: digit[i] > digit[i+1]
        # If we find any pair where digit[i] <= digit[i+1], it's not 321-like.
        if n_str[i] <= n_str[i+1]:
            # If the strictly decreasing condition is violated
            is_321_like = False
            # We can stop checking immediately as one violation is enough
            break

    # Print the final result based on whether the flag remained True
    if is_321_like:
        print("Yes")
    else:
        print("No")

# Call the solve function to run the logic
solve()