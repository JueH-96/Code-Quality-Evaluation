# Read the input string from standard input
s = input()

# Split the string by the '.' character.
# This creates a list of substrings.
# For example:
# "atcoder.jp" -> ["atcoder", "jp"]
# ".z" -> ["", "z"]
# "translate.google.com" -> ["translate", "google", "com"]
# "..........txt" -> ["", "", "", "", "", "", "", "", "", "", "txt"]
parts = s.split('.')

# The problem asks for the last substring when split by '.'.
# This corresponds to the last element in the list created by split.
# Since the string does not end with '.', the last element is guaranteed
# to be the longest suffix that does not contain '.', and is non-empty.
last_part = parts[-1]

# Print the result to standard output.
print(last_part)