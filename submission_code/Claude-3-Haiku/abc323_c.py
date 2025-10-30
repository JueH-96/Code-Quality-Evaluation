import sys

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))
S = [input() for _ in range(N)]

# Calculate the total score for each player
total_scores = [sum(A[j] for j in range(M) if S[i][j] == 'o') + i for i in range(N)]

# Solve the problem for each player
for i in range(N):
    unsolved = [j for j in range(M) if S[i][j] == 'x']
    min_problems = 0
    for j in range(N):
        if i != j and total_scores[j] >= total_scores[i]:
            min_problems = max(min_problems, (total_scores[j] - total_scores[i] + sum(A[k] for k in unsolved)) // sum(A[k] for k in unsolved))
    print(min_problems)