# Read the input string from standard input
S = input()

# Split the string S by the delimiter '.'
# This creates a list of substrings.
# For example, if S is "atcoder.jp", parts will be ['atcoder', 'jp'].
# If S is ".z", parts will be ['', 'z'].
# If S is "..........txt", parts will be ['', '', '', '', '', '', '', '', '', '', 'txt'].
parts = S.split('.')

# The problem asks for the last substring after splitting.
# This is the last element in the list 'parts'.
# The constraint that S does not end with '.' ensures that this last part
# will not be an empty string caused by a trailing delimiter.
result = parts[-1]

# Print the result to standard output
print(result)