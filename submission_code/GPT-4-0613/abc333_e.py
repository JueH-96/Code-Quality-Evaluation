import heapq
from collections import deque
import sys

N = int(input().strip())
events = [list(map(int, input().strip().split())) for _ in range(N)]

potion_events = deque(sorted((x, i) for i, (t, x) in enumerate(events) if t == 1))
monster_events = deque(sorted((x, i) for i, (t, x) in enumerate(events) if t == 2))

potion_picked = [0]*N
potion_heap = []
heapq.heapify(potion_heap)

for monster_x, monster_i in monster_events:
    while potion_events and potion_events[0][0] <= monster_x and potion_events[0][1] < monster_i:
        potion_x, potion_i = potion_events.popleft()
        heapq.heappush(potion_heap, (-potion_x, potion_i))
        potion_picked[potion_i] = 1

    if not potion_heap or potion_heap[0][1] > monster_i:
        print(-1)
        sys.exit(0)

    _, potion_i = heapq.heappop(potion_heap)
    potion_picked[potion_i] = 2

while potion_events:
    potion_x, potion_i = potion_events.popleft()
    heapq.heappush(potion_heap, (-potion_x, potion_i))
    potion_picked[potion_i] = 1

print(potion_picked.count(2))
print(" ".join(str(int(p == 2)) for p in potion_picked))