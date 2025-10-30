# YOUR CODE HERE
# Read the input string from standard input.
s = input()

# The problem asks to print the last substring when the input string S is split by '.'s.
# The string method `split('.')` separates the string into a list of substrings based on the '.' delimiter.
# For example, 'a.b.c'.split('.') would result in the list ['a', 'b', 'c'].
# We are interested in the last element of this list.
# The index [-1] in Python accesses the last element of a list.
# The problem constraints ensure that there is at least one '.' and the string doesn't end with a '.',
# so this operation is well-defined and will produce the correct result.
last_substring = s.split('.')[-1]

# Print the result to standard output.
print(last_substring)