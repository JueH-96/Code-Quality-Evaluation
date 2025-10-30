# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
A = list(map(int, input[:9]))
B = list(map(int, input[9:]))

takahashi_score = sum(A)
aoki_score = sum(B)

# Calculate the minimum runs Aoki needs to score in the bottom of the ninth to win
min_runs_needed = takahashi_score - aoki_score + 1

print(min_runs_needed)