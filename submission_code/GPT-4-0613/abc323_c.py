N, M = map(int, input().split())
A = list(map(int, input().split()))
S = [input() for _ in range(N)]

scores = [sum(A[j] if S[i][j] == 'o' else 0 for j in range(M)) + i + 1 for i in range(N)]
max_score = max(scores)

for i in range(N):
    unsolved = [A[j] for j in range(M) if S[i][j] == 'x']
    unsolved.sort(reverse=True)
    total = scores[i]
    cnt = 0
    while total < max_score:
        total += unsolved[cnt]
        cnt += 1
    print(cnt)