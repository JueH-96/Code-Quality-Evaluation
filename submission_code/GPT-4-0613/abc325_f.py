import sys
import heapq

N = int(input().strip())
D = list(map(int, input().strip().split()))
L1, C1, K1 = map(int, input().strip().split())
L2, C2, K2 = map(int, input().strip().split())

if C1 * L2 < C2 * L1:
    L1, L2 = L2, L1
    C1, C2 = C2, C1
    K1, K2 = K2, K1

D.sort(reverse=True)
q = []
res = 0
for d in D:
    while d > L1 and K2 > 0:
        d -= L2
        K2 -= 1
        res += C2
        heapq.heappush(q, L2)
    while q and q[0] < d:
        t = heapq.heappop(q)
        d -= t
        K2 += 1
        res -= C2
    if d > L1:
        print(-1)
        sys.exit()
    res += C1
    K1 -= 1
    if K1 < 0:
        t = heapq.heappop(q)
        K2 += 1
        res -= C2
        K1 += 1
        res -= C1
print(res)