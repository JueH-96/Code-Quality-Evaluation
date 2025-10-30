# Read the input string S
S = input()

# The problem asks to check characters at 1-based even positions: 2, 4, ..., 16.
# In 0-based indexing (used by Python strings), these are indices: 1, 3, ..., 15.

# We can extract these characters using string slicing: S[start:stop:step].
# S[1::2] means start at index 1, go to the end of the string, with a step of 2.
# This will select S[1], S[3], S[5], ..., S[15].
relevant_chars_substring = S[1::2]

# The condition is that all these characters must be '0'.
# This is equivalent to checking that the character '1' does not appear
# in the substring of relevant characters.
# (Since S is guaranteed to consist only of '0's and '1's)
if '1' in relevant_chars_substring:
    # If '1' is present, then not all characters are '0'.
    print("No")
else:
    # If '1' is not present, then all characters must be '0'.
    print("Yes")