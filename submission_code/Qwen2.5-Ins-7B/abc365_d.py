# N = int(input())
# S = input()
N = 6
S = "PRSSRS"

def max_wins(N, S):
    prev = None
    wins = 0
    for move in S:
        if prev is None:
            prev = move
            continue
        if (prev == 'R' and move == 'P') or (prev == 'P' and move == 'S') or (prev == 'S' and move == 'R'):
            wins += 1
        prev = move
    return wins

print(max_wins(N, S))