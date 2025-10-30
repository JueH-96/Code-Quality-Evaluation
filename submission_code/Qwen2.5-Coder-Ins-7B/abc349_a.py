# YOUR CODE HERE
import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Extract N and the list of scores for persons 1 to N-1
N = int(data[0])
A = list(map(int, data[1:]))

# Calculate the final score of person N
final_score_N = sum(A)

# Print the final score of person N
print(final_score_N)