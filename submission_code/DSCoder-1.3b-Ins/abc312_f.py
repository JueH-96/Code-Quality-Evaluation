import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    items = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
    items.sort(key=lambda x: x[1], reverse=True)
    happiness = 0
    for i in range(M):
        if items[i][0] == 0:
            happiness += items[i][1]
        elif items[i][0] == 1:
            happiness += items[i][1]
        elif items[i][0] == 2:
            happiness += items[i][1]
    print(happiness)

solve()