# YOUR CODE HERE
import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Parse the input
N = int(data[0])
X = int(data[1])
scores = list(map(int, data[2:]))

# Calculate the total score for problems with a score of X or less
total_score = sum(score for score in scores if score <= X)

# Print the result
print(total_score)