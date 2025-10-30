# YOUR CODE HERE
import sys

# Read input from standard input
S = sys.stdin.read().strip()

# Find the indices of the two '|'
first_pipe_index = S.index('|')
second_pipe_index = S.index('|', first_pipe_index + 1)

# Remove the characters between the two '|' including the '|' themselves
result = S[:first_pipe_index] + S[second_pipe_index + 1:]

# Print the result
print(result)