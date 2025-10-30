from collections import defaultdict
from heapq import *

N, T = map(int, input().split())

events = []
for _ in range(T):
    a, b = map(int, input().split())
    events.append((a, b))

events.sort()

scores = defaultdict(int)
heap = []

for a, b in events:
    heappush(heap, -b)
    scores[b] += 1
    while heap and scores[-heap[0]] == 0:
        heappop(heap)
    print(len(heap))