import sys

def solve():
    N = int(sys.stdin.readline().strip())
    slimes = [tuple(map(int, sys.stdin.readline().strip().split())) for _ in range(N)]
    slimes.sort()
    total = 0
    for s, c in slimes:
        total += c
        if total >= 2:
            total -= 1
            c -= 1
    print(total)

solve()