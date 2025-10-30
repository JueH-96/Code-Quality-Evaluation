# YOUR CODE HERE
import sys

# Read the input string from standard input
# .strip() removes potential leading/trailing whitespace, including the newline character
S = sys.stdin.readline().strip()

# Split the string S using '|' as the delimiter.
# The problem guarantees that S contains exactly two '|' characters.
# Therefore, the split operation will always result in a list containing exactly three strings.
# For example:
# If S = "atcoder|beginner|contest", parts will be ['atcoder', 'beginner', 'contest']
# If S = "|spoiler|", parts will be ['', 'spoiler', '']
# If S = "||xyz", parts will be ['', '', 'xyz']
parts = S.split('|')

# The task is to remove the characters between the two '|'s, including the '|'s themselves.
# This means we need to keep the part of the string before the first '|'
# and the part of the string after the second '|'.
# In the `parts` list:
# - parts[0] contains the substring before the first '|'.
# - parts[1] contains the substring between the two '|'s (which we need to remove).
# - parts[2] contains the substring after the second '|'.
# We concatenate parts[0] and parts[2] to get the desired result.
result = parts[0] + parts[2]

# Print the resulting string to standard output.
print(result)