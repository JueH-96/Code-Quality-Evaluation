import sys
from heapq import heappush, heappop

input = sys.stdin.read
data = input().split()

n = int(data[0])
m = int(data[1])

pull_tab_cans = []
regular_cans = []
can_openers = []

for i in range(n):
    t = int(data[2 * i + 2])
    x = int(data[2 * i + 3])
    if t == 0:
        pull_tab_cans.append(x)
    elif t == 1:
        regular_cans.append(x)
    else:
        can_openers.append(x)

pull_tab_cans.sort(reverse=True)
regular_cans.sort(reverse=True)
can_openers.sort(reverse=True)

max_happiness = 0
used_openers = 0

# Use pull-tab cans first
for i in range(min(m, len(pull_tab_cans))):
    max_happiness += pull_tab_cans[i]
    m -= 1

# Use regular cans and can openers
for i in range(min(m, len(regular_cans))):
    if used_openers < len(can_openers) and can_openers[used_openers] > 0:
        max_happiness += regular_cans[i]
        can_openers[used_openers] -= 1
        if can_openers[used_openers] == 0:
            used_openers += 1
    else:
        break
    m -= 1

print(max_happiness)