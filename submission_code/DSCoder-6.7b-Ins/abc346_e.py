import sys
from collections import defaultdict

def read_input():
    H, W, M = map(int, sys.stdin.readline().split())
    operations = [list(map(int, sys.stdin.readline().split())) for _ in range(M)]
    return H, W, M, operations

def solve(H, W, M, operations):
    rows = defaultdict(set)
    cols = defaultdict(set)
    colors = defaultdict(int)

    for T, A, X in operations:
        if T == 1:
            rows[A].add(X)
        else:
            cols[A].add(X)

    for i in range(1, H+1):
        for color in rows[i]:
            colors[color] += 1

    for i in range(1, W+1):
        for color in cols[i]:
            colors[color] += 1

    colors = sorted((v, k) for k, v in colors.items())

    print(len(colors))
    for count, color in colors:
        print(color, count)

if __name__ == "__main__":
    H, W, M, operations = read_input()
    solve(H, W, M, operations)