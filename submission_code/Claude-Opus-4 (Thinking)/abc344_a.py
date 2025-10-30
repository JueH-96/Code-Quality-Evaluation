# YOUR CODE HERE
S = input()

# Find the indices of the two |s
first_pipe = S.index('|')
second_pipe = S.index('|', first_pipe + 1)

# Concatenate the parts before the first | and after the second |
result = S[:first_pipe] + S[second_pipe + 1:]

print(result)