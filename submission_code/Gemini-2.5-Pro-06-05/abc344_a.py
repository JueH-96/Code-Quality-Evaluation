# YOUR CODE HERE
# Read the input string from standard input.
S = input()

# The string S is guaranteed to contain exactly two '|' characters.
# We can use the split() method to divide the string into three parts
# using '|' as the delimiter.
# For example, "part1|part2|part3" becomes ['part1', 'part2', 'part3'].
parts = S.split('|')

# The problem requires removing the characters between the two '|'s,
# including the '|'s themselves. This is equivalent to keeping the first
# part (before the first '|') and the third part (after the second '|').
# These correspond to the elements at index 0 and 2 of the 'parts' list.
result = parts[0] + parts[2]

# Print the resulting string to standard output.
print(result)