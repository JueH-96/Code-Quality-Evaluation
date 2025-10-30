import sys

def solve():
    N = int(input())
    buildings = []
    for _ in range(N):
        x, h = map(int, input().split())
        buildings.append((x, h))

    buildings.sort(key=lambda x: x[0])

    max_height = 0
    for i in range(N):
        min_height = float('inf')
        for j in range(i):
            min_height = min(min_height, buildings[j][1])
        max_height = max(max_height, buildings[i][1] - min_height)

    if max_height == 0:
        print(-1)
    else:
        print(max_height)

solve()