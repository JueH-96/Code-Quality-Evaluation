# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
scores = list(map(int, input[1:]))

# The sum of all scores must be zero because each win/loss transaction
# adds a +1 and a -1 to the total score.
total_score = sum(scores)
score_N = -total_score

print(score_N)