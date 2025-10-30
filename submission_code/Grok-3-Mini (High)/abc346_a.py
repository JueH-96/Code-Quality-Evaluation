import sys
data = sys.stdin.read().split()
N = int(data[0])
A = list(map(int, data[1: N+1]))
print(' '.join(str(A[i] * A[i+1]) for i in range(N-1)))