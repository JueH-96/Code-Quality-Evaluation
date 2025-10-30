from collections import defaultdict

def solve():
    N = int(input())
    results = [input() for _ in range(N)]

    # Count wins for each player
    wins = [0] * N
    for i in range(N):
        for j in range(N):
            if i != j and results[i][j] == 'o':
                wins[i] += 1

    # Sort players by wins, then by player number
    players = sorted(range(1, N+1), key=lambda x: (-wins[x-1], x))

    # Print the player numbers in descending order of rank
    print(' '.join(map(str, players)))

solve()