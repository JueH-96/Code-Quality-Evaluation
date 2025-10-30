import sys

def solve():
    N = int(sys.stdin.readline().strip())
    wins = [0] * (N+1)
    players = list(range(1, N+1))
    for i in range(1, N+1):
        for j, result in enumerate(sys.stdin.readline().strip(), 1):
            if i != j and result == 'o':
                wins[i] += 1
    players.sort(key=lambda x: (-wins[x], x))
    print(' '.join(map(str, players)))

solve()