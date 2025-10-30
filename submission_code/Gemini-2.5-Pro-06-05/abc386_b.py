import sys

# Read the input string from standard input.
# .strip() removes any leading/trailing whitespace, including the newline character.
S = sys.stdin.readline().strip()

# The total number of presses can be calculated based on a simple observation.
# The maximum number of presses would be the length of the string, if we pressed
# one button for each character.
# The '00' button lets us type two characters ('0', '0') with a single press,
# which saves one press compared to pressing the '0' button twice.
# To minimize the total presses, we should maximize this saving by using the
# '00' button whenever possible.

# The Python str.count() method finds the number of non-overlapping
# occurrences of a substring. This is exactly what a greedy left-to-right
# scan for "00" would find.
# Each "00" found represents a saving of 1 press.
savings = S.count("00")

# The minimum number of presses is the length of the string minus the savings.
# Let k = S.count("00"). These are k presses for 2*k characters.
# The other len(S) - 2*k characters require one press each.
# Total presses = k + (len(S) - 2*k) = len(S) - k.
min_presses = len(S) - savings

# Print the result to standard output.
print(min_presses)