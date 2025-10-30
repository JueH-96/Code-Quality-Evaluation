import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def solve():
    n = int(input())
    checkpoints = []
    for _ in range(n):
        x, y = map(int, input().split())
        checkpoints.append((x, y))

    min_s = float('inf')

    for i in range(2**(n-2)):
        skipped_count = 0
        current_path = [0]
        binary_representation = bin(i)[2:].zfill(n-2)

        for j in range(n-2):
            if binary_representation[j] == '0':
                current_path.append(j+1)
            else:
                skipped_count += 1

        current_path.append(n-1)

        total_distance = 0
        for k in range(len(current_path) - 1):
            total_distance += distance(checkpoints[current_path[k]][0], checkpoints[current_path[k]][1],
                                        checkpoints[current_path[k+1]][0], checkpoints[current_path[k+1]][1])

        if skipped_count > 0:
            penalty = 2**(skipped_count - 1)
        else:
            penalty = 0

        min_s = min(min_s, total_distance + penalty)

    print(min_s)

solve()