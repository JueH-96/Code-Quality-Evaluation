# YOUR CODE HERE
S = input().strip()

# Find the positions of the two |s
first_pipe = S.index('|')
second_pipe = S.index('|', first_pipe + 1)

# Concatenate the part before first | and the part after second |
result = S[:first_pipe] + S[second_pipe + 1:]

print(result)