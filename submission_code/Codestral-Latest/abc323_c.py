import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

A = [int(data[index + i]) for i in range(M)]
index += M

S = [data[index + i] for i in range(N)]
index += N

scores = []
for i in range(N):
    score = i
    for j in range(M):
        if S[i][j] == 'o':
            score += A[j]
    scores.append(score)

for i in range(N):
    max_score = 0
    for j in range(N):
        if i != j:
            max_score = max(max_score, scores[j])

    unsolved = [A[j] for j in range(M) if S[i][j] == 'x']
    unsolved.sort(reverse=True)

    total = scores[i]
    count = 0
    for score in unsolved:
        total += score
        count += 1
        if total > max_score:
            break

    print(count)