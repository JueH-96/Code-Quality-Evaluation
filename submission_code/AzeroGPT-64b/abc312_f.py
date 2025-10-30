import heapq
from collections import deque, defaultdict
from itertools import accumulate
N, M = map(int, input().split())
items = [list(map(int, input().split())) for _ in range(N)]

DRAW = 0
PULL = 0
REGULAR = 1
OPENER = 2

sieve = defaultdict(list)
tmp = []
for t, x in items:
    if t == PULL or t == DRAW:
        tmp.append(x)
    else:
        sieve[t-1].append(x)

pull, regular, opener = tmp, sieve[REGULAR], sieve[OPENER]
acc = list(accumulate(sorted(pull, reverse=True), max))

ans = 0
heap = []
for x in sorted(regular, reverse=True):
    heapq.heappush(heap, x)
for x in sorted(opener, reverse=True):
    while x > 0 and heap:
        ans += heapq.heappop(heap)
        x -= 1
    if heap:
        heapq.heappush(heap, 0)
    M -= 1

ans += sum(acc[:M])
print(ans)