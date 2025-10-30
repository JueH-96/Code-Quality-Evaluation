# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1:N+1]
M = max(len(s) for s in S)

T = [''] * M

for i in range(N):
    for j in range(len(S[i])):
        T[j] += S[i][j]
    for j in range(len(S[i]), M):
        T[j] += '*'

for i in range(M):
    T[i] = T[i].rstrip('*')

for t in T:
    print(t)