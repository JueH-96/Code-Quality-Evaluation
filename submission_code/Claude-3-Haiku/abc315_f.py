import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def solve():
    N = int(input())
    checkpoints = []
    for _ in range(N):
        x, y = map(int, input().split())
        checkpoints.append((x, y))

    min_distance = float('inf')
    for skip in range(1, 2 ** (N - 2)):
        total_distance = 0
        skipped = 0
        x, y = checkpoints[0]
        for i in range(1, N):
            if (skip >> (i - 1)) & 1:
                skipped += 1
                continue
            x2, y2 = checkpoints[i]
            total_distance += distance(x, y, x2, y2)
            x, y = x2, y2
        if skipped > 0:
            total_distance += 2 ** (skipped - 1)
        min_distance = min(min_distance, total_distance)

    print(f"{min_distance:.15f}")

solve()