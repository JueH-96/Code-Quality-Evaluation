# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
A = list(map(int, input[1:]))

B = [A[i] * A[i+1] for i in range(N-1)]

print(" ".join(map(str, B)))