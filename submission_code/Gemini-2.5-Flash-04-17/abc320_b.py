# YOUR CODE HERE
import sys

# Read the input string from standard input.
# .strip() removes any leading or trailing whitespace, including the newline character.
S = sys.stdin.readline().strip()

# Get the length of the input string.
n = len(S)

# Iterate through possible lengths of palindromic substrings.
# We start from the maximum possible length (n) and go down to 1.
# This is because we want to find the maximum length; the first palindrome found
# in this decreasing search will have the maximum length.
for length in range(n, 0, -1):
    # For the current 'length', iterate through all possible starting indices 'i'.
    # A substring of length 'length' starting at index 'i' is S[i : i + length].
    # The valid range for 'i' is such that the entire substring is within S.
    # The end index i + length - 1 must be less than n. So, i <= n - length.
    # Thus, 'i' goes from 0 up to n - length.
    for i in range(n - length + 1):
        # Extract the contiguous substring using slicing.
        substring = S[i : i + length]

        # Check if the extracted substring is a palindrome.
        # A string is a palindrome if it reads the same forwards and backwards.
        # In Python, slicing [::-1] creates a reversed copy of the string.
        if substring == substring[::-1]:
            # If the substring is a palindrome, its length 'length' is the
            # maximum length we could find because we are checking lengths
            # in decreasing order.
            print(length)
            # We have found the answer, so we can terminate the program early.
            sys.exit()

# The problem guarantees that at least one contiguous palindromic substring exists
# in S (any single character is a palindrome). Thus, the loops will always find
# a palindrome (at least of length 1) and the program will exit via sys.exit().
# This ensures that the program will always print an answer and terminate,
# making the code after the loop effectively unreachable.