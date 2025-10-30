import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    scores = list(map(int, sys.stdin.readline().split()))
    players = [sys.stdin.readline().strip() for _ in range(N)]

    total_scores = [0] * N
    for i in range(N):
        for j in range(M):
            if players[i][j] == 'o':
                total_scores[i] += scores[j]
        total_scores[i] += i

    total_scores.sort()

    for i in range(N):
        remaining_problems = sum(scores[j] for j in range(M) if players[i][j] == 'x')
        remaining_points = total_scores[i] - total_scores[0]
        remaining_players = N - i - 1

        if remaining_points + remaining_problems * (remaining_players + 1) > total_scores[0]:
            print(i)
            return

    print(N)

solve()