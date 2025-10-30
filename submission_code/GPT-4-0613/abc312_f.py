import heapq
import sys

N, M = map(int, input().split())
pull_tab_cans = []
regular_cans = []
can_openers = []
for _ in range(N):
    T, X = map(int, input().split())
    if T == 0:
        pull_tab_cans.append(X)
    elif T == 1:
        regular_cans.append(X)
    else:
        can_openers.append(X)

pull_tab_cans.sort(reverse=True)
regular_cans.sort(reverse=True)
can_openers.sort(reverse=True)

happiness = 0
used_can_openers = 0
while M > 0:
    if pull_tab_cans and (not regular_cans or not can_openers or pull_tab_cans[0] > regular_cans[0]):
        happiness += pull_tab_cans.pop(0)
    elif regular_cans and can_openers and can_openers[0] > 0:
        happiness += regular_cans.pop(0)
        can_openers[0] -= 1
        used_can_openers += 1
    elif can_openers:
        can_openers.pop(0)
    M -= 1

print(happiness)