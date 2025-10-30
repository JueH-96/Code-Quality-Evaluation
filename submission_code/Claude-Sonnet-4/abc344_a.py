# YOUR CODE HERE
S = input().strip()

# Find positions of the two | characters
first_pipe = S.find('|')
second_pipe = S.find('|', first_pipe + 1)

# Get the part before first | and the part after second |
result = S[:first_pipe] + S[second_pipe + 1:]

print(result)