import sys
from collections import defaultdict

def min_cost():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))
    W = list(map(int, sys.stdin.readline().strip().split()))

    box_items = defaultdict(list)
    for i in range(N):
        box_items[A[i]].append(W[i])

    for box in box_items:
        box_items[box].sort()

    total_cost = 0
    for box in range(1, N+1):
        if box in box_items:
            for i in range(len(box_items[box])-1):
                total_cost += box_items[box][i]
        else:
            total_cost += sum(box_items[i])

    return total_cost

print(min_cost())