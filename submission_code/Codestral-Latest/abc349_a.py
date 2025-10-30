# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# The final score of person N is the negative sum of the scores of persons 1 to N-1
score_N = -sum(A)

print(score_N)