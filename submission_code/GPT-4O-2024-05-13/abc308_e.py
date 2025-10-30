# YOUR CODE HERE
def mex(a, b, c):
    if 0 not in (a, b, c):
        return 0
    if 1 not in (a, b, c):
        return 1
    if 2 not in (a, b, c):
        return 2
    return 3

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
S = data[N+1]

sum_mex = 0

for i in range(N):
    if S[i] == 'M':
        for j in range(i+1, N):
            if S[j] == 'E':
                for k in range(j+1, N):
                    if S[k] == 'X':
                        sum_mex += mex(A[i], A[j], A[k])

print(sum_mex)