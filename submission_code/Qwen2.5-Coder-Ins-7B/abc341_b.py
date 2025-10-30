# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
S = []
T = []
for i in range(N-1):
    S.append(int(data[N+2*i+1]))
    T.append(int(data[N+2*i+2]))

for i in range(N-1):
    while A[i] >= S[i]:
        A[i] -= S[i]
        A[i+1] += T[i]

print(A[-1])