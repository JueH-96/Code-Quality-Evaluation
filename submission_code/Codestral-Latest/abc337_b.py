# YOUR CODE HERE
import sys

def is_extended_abc_string(s):
    if not s:
        return True

    # Find the first occurrence of 'B'
    b_index = s.find('B')
    if b_index == -1:
        # If 'B' is not found, check if the string is all 'A's or all 'C's
        return s == 'A' * len(s) or s == 'C' * len(s)

    # Find the first occurrence of 'C' after 'B'
    c_index = s.find('C', b_index)
    if c_index == -1:
        # If 'C' is not found, check if the string is all 'A's or all 'B's
        return s == 'A' * len(s) or s == 'B' * len(s)

    # Check if the string is a concatenation of 'A's, 'B's, and 'C's
    return s[:b_index] == 'A' * b_index and s[b_index:c_index] == 'B' * (c_index - b_index) and s[c_index:] == 'C' * (len(s) - c_index)

# Read input from stdin
input_string = sys.stdin.read().strip()

# Check if the input string is an Extended ABC string
if is_extended_abc_string(input_string):
    print("Yes")
else:
    print("No")