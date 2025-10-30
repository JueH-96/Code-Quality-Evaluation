N, M = map(int, input().split())
A = list(map(int, input().split()))
A.sort(reverse=True)
S = [input() for _ in range(N)]

score = [0] * N
for i in range(N):
    for j in range(M):
        if S[i][j] == 'o':
            score[i] += A[j]

need = [0] * N
for i in range(N):
    for j in range(M):
        if S[i][j] == 'x':
            for k in range(N):
                if score[i] + j + 1 > score[k]:
                    need[i] = j + 1
                    break

for i in range(N):
    print(need[i])