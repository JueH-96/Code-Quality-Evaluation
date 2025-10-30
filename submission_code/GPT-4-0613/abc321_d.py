import heapq

N, M, P = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

A.sort()
B.sort()

heap = []
for a in A:
    while B and a + B[-1] > P:
        heapq.heappush(heap, a + B.pop())
    if B:
        heapq.heappush(heap, a + B[-1])

total = 0
for _ in range(N * M):
    total += heapq.heappop(heap)

print(total)