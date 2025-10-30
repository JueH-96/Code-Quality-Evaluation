# Read the input from stdin
import sys

# Read the number of people and their scores
N, *A = map(int, sys.stdin.readline().split())

# Calculate the final score of person N
final_score = -sum(A)

# Print the final score
print(final_score)