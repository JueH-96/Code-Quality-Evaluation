import heapq

N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

AB = []
for i in range(N):
    AB.append((A[i], A[i]))

for i in range(M):
    AB.append((B[i], 10**9+1))

AB.sort()

heap = []
heapq.heapify(heap)

ans = 0
for a, b in AB:
    if b == 10**9+1:
        if len(heap) == 0 or heap[0] < a:
            print(-1)
            exit()
        ans += heapq.heappop(heap)
    else:
        heapq.heappush(heap, b)

print(ans)