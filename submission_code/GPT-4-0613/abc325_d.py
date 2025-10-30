import heapq
import sys

N = int(input().strip())
TD = [list(map(int, input().strip().split())) for _ in range(N)]
TD.sort(key=lambda x: (x[0], -x[1]))

heap = []
ans = 0
for t, d in TD:
    while heap and heap[0] < t:
        heapq.heappop(heap)
    heapq.heappush(heap, d)
    ans = max(ans, len(heap))
print(ans)