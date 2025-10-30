# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))

B = [A[i] * A[i+1] for i in range(N-1)]

print(' '.join(map(str, B)))