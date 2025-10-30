# YOUR CODE HERE
from heapq import heappop, heappush

N, M = map(int, input().split())
P = list(map(int, input().split()))
L, D = [], []
for _ in range(M):
    l, d = map(int, input().split())
    L.append(l)
    D.append(d)

P.sort()
coupons = sorted([(L[i], D[i]) for i in range(M)])

total = 0
j = 0
heap = []
for i in range(N):
    while j < M and coupons[j][0] <= P[i]:
        heappush(heap, -coupons[j][1])
        j += 1
    if heap:
        total += heappop(heap)
    else:
        total += P[i]

print(abs(total))