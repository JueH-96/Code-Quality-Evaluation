import heapq

N, K = map(int, input().split())
medicines = [list(map(int, input().split())) for _ in range(N)]

heap = []
for a, b in medicines:
    heapq.heappush(heap, (a, b))

total = 0
while heap:
    a, b = heapq.heappop(heap)
    total += b
    if total > K:
        total -= b
        heapq.heappush(heap, (a + 1, b))
    else:
        print(a)
        break