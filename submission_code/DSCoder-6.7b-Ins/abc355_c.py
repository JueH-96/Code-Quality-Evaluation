import sys
from collections import defaultdict

def solve():
    N, T = map(int, sys.stdin.readline().split())
    announced = list(map(int, sys.stdin.readline().split()))

    rows = defaultdict(set)
    cols = defaultdict(set)
    diag1 = set()
    diag2 = set()

    for turn, num in enumerate(announced, start=1):
        row, col = divmod(num-1, N)
        rows[row].add(turn)
        cols[col].add(turn)
        if row == col:
            diag1.add(turn)
        if row + col == N - 1:
            diag2.add(turn)
        if len(rows[row]) == N or len(cols[col]) == N or len(diag1) == N or len(diag2) == N:
            return turn
    return -1

print(solve())