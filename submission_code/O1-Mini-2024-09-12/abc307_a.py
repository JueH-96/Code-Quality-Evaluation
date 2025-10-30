import sys

N,*A = map(int, sys.stdin.read().split())
sums = [sum(A[7*i:7*(i+1)]) for i in range(N)]
print(' '.join(map(str, sums)))