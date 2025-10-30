# YOUR CODE HERE
import sys

def solve():
    """
    Reads a string S from standard input, checks if it represents a valid
    past AtCoder contest abbreviation based on the given rules, and prints
    "Yes" or "No" to standard output.
    """
    # Read the input string from standard input
    s = sys.stdin.readline().strip()

    # Extract the numeric part of the string.
    # The constraints guarantee that S has length 6, starts with 'ABC',
    # and the last three characters are digits.
    numeric_part = s[3:]

    # Convert the numeric part to an integer.
    # Since the constraints guarantee the last three characters are digits,
    # this conversion will always succeed.
    n = int(numeric_part)

    # Check if the number n satisfies the conditions:
    # 1. n must be between 1 and 349 (inclusive).
    # 2. n must not be equal to 316.
    if 1 <= n <= 349 and n != 316:
        # If both conditions are met, it's a valid past contest abbreviation.
        print("Yes")
    else:
        # Otherwise, it's not a valid past contest abbreviation.
        print("No")

# Call the solve function to execute the logic
solve()