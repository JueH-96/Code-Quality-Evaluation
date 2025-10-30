# Read the input string from standard input
S = input()

# Find the indices of the two | characters
first_pipe = S.find('|')
second_pipe = S.rfind('|')

# Remove the characters between the two |s, including the |s themselves
result = S[:first_pipe] + S[second_pipe+1:]

# Print the resulting string
print(result)