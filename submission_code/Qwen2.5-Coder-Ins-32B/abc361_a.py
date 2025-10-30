# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
K = int(input[1])
X = int(input[2])
A = list(map(int, input[3:]))

B = A[:K] + [X] + A[K:]

print(" ".join(map(str, B)))