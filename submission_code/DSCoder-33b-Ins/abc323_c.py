# YOUR CODE HERE

N, M = map(int, input().split())
A = list(map(int, input().split()))

players = []
for i in range(N):
    players.append(input())

scores = [0]*N
for i in range(N):
    for j in range(M):
        if players[i][j] == 'o':
            scores[i] += A[j]
    scores[i] += i+1

max_score = max(scores)
answers = []
for i in range(N):
    needed_score = max_score - scores[i] + 1
    needed_problems = needed_score // 100
    answers.append(needed_problems)

for answer in answers:
    print(answer)