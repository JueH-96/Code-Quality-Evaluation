# YOUR CODE HERE
from heapq import heappush, heappop

N, M = map(int, input().split())
A = sorted([int(x) for x in input().split()])
B = sorted([int(x) for x in input().split()])

i, j = 0, 0
heap = []
ans = 0
while i < N and j < M:
    if A[i] >= B[j]:
        ans += A[i]
        j += 1
    heappush(heap, -A[i])
    i += 1

while j < M and heap:
    ans += -heappop(heap)
    j += 1

if j == M:
    print(ans)
else:
    print(-1)