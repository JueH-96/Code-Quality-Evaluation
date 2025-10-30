import sys

# Read input
line = sys.stdin.readline().strip().split()
N, M = int(line[0]), int(line[1])

S = []
for _ in range(N):
    S.append(list(sys.stdin.readline().strip()))

T = []
for _ in range(M):
    T.append(list(sys.stdin.readline().strip()))

# Find the location of T in S
for a in range(N - M + 1):
    for b in range(N - M + 1):
        if all(S[a+i][b+j] == T[i][j] for i in range(M) for j in range(M)):
            print(a+1, b+1)
            break