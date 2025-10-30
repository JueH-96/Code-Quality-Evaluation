# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
K = int(data[1])
X = int(data[2])
A = list(map(int, data[3:]))

B = A[:K] + [X] + A[K:]
print(' '.join(map(str, B)))