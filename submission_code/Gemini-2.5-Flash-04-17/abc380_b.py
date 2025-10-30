# YOUR CODE HERE
import sys

# Read the input string from stdin
s = sys.stdin.readline().strip()

# The string S is generated according to the process:
# Start with S = "|".
# For i = 1 to N, append A_i copies of '-' and then append "|".
# This means S has the structure:
# "|" + (A_1 * "-" + "|") + (A_2 * "-" + "|") + ... + (A_N * "-" + "|")

# If we remove the initial "|", we are left with the sequence of blocks:
# (A_1 * "-" + "|") + (A_2 * "-" + "|") + ... + (A_N * "-" + "|")
s_without_first_bar = s[1:]

# Now, if we split this string 's_without_first_bar' by "|",
# each segment *before* a "|" corresponds to a block of A_i hyphens.
# Since the string ends with "|", the split will result in an empty string
# as the last element in the list.
# Example: If S = "|--|---|", then s_without_first_bar = "--|---|".
# "--|---|".split('|') gives the list ['--', '---', ''].
segments = s_without_first_bar.split('|')

# The segments of hyphens corresponding to A_1, A_2, ..., A_N are
# all the elements in the 'segments' list except the last one (which is the empty string).
# The length of each segment of hyphens (e.g., "--") is the value of the corresponding A_i.
# We use a list comprehension to calculate the length for each segment
# in the list, excluding the last element (segments[:-1]).
a_values = [len(segment) for segment in segments[:-1]]

# The problem asks to print the elements of the reconstructed sequence A
# separated by spaces on a single line.
# We convert the list of integers (a_values) into a list of strings using map(str, ...).
# Then, we join these strings together using a space (' ') as the separator.
print(' '.join(map(str, a_values)))