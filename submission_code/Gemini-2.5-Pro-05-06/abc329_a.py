# Read the input string S from standard input
S = input()

# Use the str.join() method to insert a space between each character of S.
# A string is an iterable of its characters. For example, if S is "ABC",
# " ".join(S) will result in the string "A B C".
result_string = " ".join(S)

# Print the resulting string to standard output.
# The print() function by default adds a newline at the end of the output.
print(result_string)