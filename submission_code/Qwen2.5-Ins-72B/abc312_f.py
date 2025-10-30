# YOUR CODE HERE
import sys
from heapq import heappush, heappop

def max_happiness(N, M, items):
    pull_tabs = []
    regular_cans = []
    can_openers = []

    for T, X in items:
        if T == 0:
            pull_tabs.append(X)
        elif T == 1:
            regular_cans.append(X)
        else:
            can_openers.append(X)

    pull_tabs.sort(reverse=True)
    regular_cans.sort(reverse=True)
    can_openers.sort(reverse=True)

    total_happiness = 0
    used_openers = 0
    can_opener_heap = []

    for i in range(M):
        if pull_tabs:
            total_happiness += pull_tabs.pop(0)
        elif can_openers and used_openers < M:
            used_openers += 1
            heappush(can_opener_heap, -can_openers.pop(0))
        elif regular_cans and can_opener_heap:
            total_happiness += regular_cans.pop(0)
            heappop(can_opener_heap)
        else:
            break

    return total_happiness

input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
items = [(int(data[2 * i + 2]), int(data[2 * i + 3])) for i in range(N)]

print(max_happiness(N, M, items))