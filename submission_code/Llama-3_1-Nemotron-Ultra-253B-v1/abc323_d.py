import heapq
from collections import defaultdict

n = int(input())
counts = defaultdict(int)
for _ in range(n):
    s, c = map(int, input().split())
    counts[s] = c

heap = list(counts.keys())
heapq.heapify(heap)

while heap:
    s = heapq.heappop(heap)
    if counts[s] == 0:
        continue
    merges = counts[s] // 2
    remaining = counts[s] % 2
    counts[s] = remaining
    new_s = s * 2
    counts[new_s] += merges
    heapq.heappush(heap, new_s)

print(sum(counts.values()))