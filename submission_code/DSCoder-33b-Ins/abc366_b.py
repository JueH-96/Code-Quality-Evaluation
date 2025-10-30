# YOUR CODE HERE

N = int(input())
S = [input() for _ in range(N)]

M = max(len(s) for s in S)
T = [['*'] * N for _ in range(M)]

for i in range(N):
    for j in range(len(S[i])):
        T[j][N-i-1] = S[i][j]

for t in T:
    while t and t[-1] == '*':
        t.pop()
    print(''.join(t))