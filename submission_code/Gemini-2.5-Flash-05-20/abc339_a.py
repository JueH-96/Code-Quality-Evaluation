# Read the input string from standard input
S = input()

# Split the string S by the '.' character.
# This method returns a list of substrings.
# For example, "atcoder.jp".split('.') would yield ['atcoder', 'jp'].
# ".z".split('.') would yield ['', 'z'].
# "..........txt".split('.') would yield ['', '', ..., '', 'txt'].
parts = S.split('.')

# The problem asks for the "last substring when S is split by .s",
# which is equivalent to the longest suffix of S that does not contain '.'.
# This corresponds to the last element in the list of parts.
# Python lists can be indexed with -1 to get the last element.
result = parts[-1]

# Print the obtained result to standard output
print(result)