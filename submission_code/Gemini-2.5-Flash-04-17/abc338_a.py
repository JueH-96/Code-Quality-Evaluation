# YOUR CODE HERE
import sys

S = sys.stdin.readline().strip()

# Check if the first character is uppercase
is_first_char_upper = S[0].isupper()

# Check if all other characters are lowercase
# S[1:] gives the substring from the second character to the end.
# If the string S has only one character, S[1:] is an empty string "".
# The built-in function all() returns True for an empty iterable.
# This correctly handles the case where S has length 1, as there are no "other characters",
# so the condition "all other characters are lowercase" is vacuously true.
are_rest_chars_lower = all(c.islower() for c in S[1:])

# The condition is satisfied if the first character is uppercase AND
# all subsequent characters are lowercase.
if is_first_char_upper and are_rest_chars_lower:
    print("Yes")
else:
    print("No")