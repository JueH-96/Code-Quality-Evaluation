# YOUR CODE HERE
import sys

# Read input from stdin
input_data = sys.stdin.read().splitlines()

# Parse the input
A = list(map(int, input_data[0].split()))
B = list(map(int, input_data[1].split()))

# Calculate the scores
takahashi_score = sum(A)
aoki_score = sum(B)

# Calculate the minimum runs Aoki needs to win
min_runs_to_win = takahashi_score - aoki_score + 1

# Output the result
print(min_runs_to_win)