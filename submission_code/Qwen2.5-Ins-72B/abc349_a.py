# YOUR CODE HERE
import sys

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N]))

# Calculate the final score of person N
final_score_N = -sum(A)

# Print the result
print(final_score_N)