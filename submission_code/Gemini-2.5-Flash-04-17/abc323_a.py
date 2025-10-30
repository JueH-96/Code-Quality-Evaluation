# YOUR CODE HERE
import sys

# Read the input string S from standard input.
# The .strip() method removes leading/trailing whitespace, including the newline character.
S = sys.stdin.readline().strip()

# The problem asks to check the characters at specific positions in the string.
# The positions are the i-th character for every even number i from 2 through 16.
# In 1-based indexing, these positions are 2, 4, 6, 8, 10, 12, 14, 16.
# In Python's 0-based indexing, these correspond to indices i-1:
# Index 1 (for 2nd character), Index 3 (for 4th character), ..., Index 15 (for 16th character).
# These are the odd indices within the string S (which has length 16, indices 0-15).

# We can extract the characters at these indices using string slicing with a step.
# The slice S[1::2] starts at index 1, goes to the end of the string, and takes every 2nd character.
# This precisely selects characters at indices 1, 3, 5, 7, 9, 11, 13, 15.

# The condition is that ALL these characters must be '0'.
# This means the slice S[1::2] must be equal to a string consisting only of '0's.
# The number of indices checked is 8 (1, 3, ..., 15).
# So, the expected slice is a string of 8 zeros ('0' * 8).

# Check if the extracted slice is equal to '00000000'.
if S[1::2] == '0' * 8:
    # If they are equal, print "Yes".
    print("Yes")
else:
    # Otherwise (if any character in the slice is '1'), print "No".
    print("No")