# Read the input string
S = input().strip()

# Find the indices of the two '|' characters
first_pipe = S.find('|')
second_pipe = S.find('|', first_pipe + 1)

# Construct the resulting string by concatenating the parts before the first pipe and after the second pipe
result = S[:first_pipe] + S[second_pipe+1:]

# Print the result
print(result)