# YOUR CODE HERE
# Read the first line containing N, c_1, and c_2
# input().split() reads a line from standard input, splits it by whitespace,
# and returns a list of strings.
# We unpack the list into n_str, c1, and c2. The value of N (n_str) is not
# needed for the logic in Python, so we can effectively ignore it.
n_str, c1, c2 = input().split()

# Read the second line containing the string S from standard input
s = input()

# Create the new string using a list comprehension
# This iterates through each character ('char') in the input string 's'.
# For each 'char', it applies a conditional expression:
# - If 'char' is NOT equal to 'c1' (char != c1), the result for this character
#   in the new list will be 'c2'.
# - If 'char' IS equal to 'c1' (else), the result will be 'char' itself
#   (the original character is kept).
# This generates a list of characters for the new string.
# Finally, ''.join() concatenates the characters in the list into a single string.
modified_s = ''.join([c2 if char != c1 else char for char in s])

# Print the resulting modified string to standard output
print(modified_s)