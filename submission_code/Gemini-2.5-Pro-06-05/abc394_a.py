# Read the input string from standard input.
S = input()

# This one-liner accomplishes the task efficiently and idiomatically in Python.
# 1. (char for char in S if char == '2') is a generator expression.
#    It iterates through the input string `S` character by character.
#    For each character, it checks if it is '2'.
#    If it is, the character is yielded.
# 2. "".join(...) takes an iterable (like our generator) and concatenates
#    all its items into a single string.
# The result is a new string containing only the '2's from the original
# string, in their original order.
result = "".join(char for char in S if char == '2')

# Print the final string to standard output.
print(result)