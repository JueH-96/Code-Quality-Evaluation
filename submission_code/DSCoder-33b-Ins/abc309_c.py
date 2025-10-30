# YOUR CODE HERE
import heapq

N, K = map(int, input().split())
medicines = []
for _ in range(N):
    a, b = map(int, input().split())
    medicines.append((a, b))

medicines.sort()

heap = []
total = 0
for a, b in medicines:
    heapq.heappush(heap, -b)
    total += b
    if total > K:
        total += heapq.heappop(heap)

print(len(heap))