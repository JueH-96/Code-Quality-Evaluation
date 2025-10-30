import heapq

N, D, P = map(int, input().split())
F = list(map(int, input().split()))

F.sort(reverse=True)
heap = []
res = 0
for i in range(N):
    if len(heap) < D:
        heapq.heappush(heap, F[i])
        res += F[i]
    else:
        if F[i] < heap[0]:
            res += F[i]
        else:
            res += P
            res -= heapq.heappop(heap)
            heapq.heappush(heap, F[i])
while len(heap) > 0 and heap[0] > P:
    res += P - heapq.heappop(heap)
print(res)