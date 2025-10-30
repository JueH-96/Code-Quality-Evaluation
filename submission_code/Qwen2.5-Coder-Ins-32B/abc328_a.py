# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
X = int(input[1])
scores = list(map(int, input[2:]))

total_score = sum(score for score in scores if score <= X)

print(total_score)