# YOUR CODE HERE
import sys

# Read input
A = list(map(int, sys.stdin.readline().strip().split()))
B = list(map(int, sys.stdin.readline().strip().split()))

# Calculate the total runs scored by each team
takahashi_score = sum(A)
aoki_score = sum(B)

# Calculate the minimum runs Team Aoki needs to score in the bottom of the ninth to win
min_runs_needed = takahashi_score - aoki_score + 1

# Output the result
print(min_runs_needed)