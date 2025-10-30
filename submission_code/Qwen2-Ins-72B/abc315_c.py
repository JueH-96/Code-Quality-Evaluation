from heapq import heappush, heappop

N = int(input())
A = [list(map(int, input().split())) for _ in range(N)]
A.sort()

# (flavor, deliciousness)
hq1 = []
hq2 = []
ans = 0
s = 0
for i in range(N):
    f, s = A[i]
    if i == 0 or f != A[i-1][0]:
        if hq1:
            ans += heappop(hq1)
        if hq2:
            ans += heappop(hq2)
    heappush(hq1, -s)
    heappush(hq2, -s)
    if len(hq1) > 2:
        heappush(hq2, heappop(hq1))
    if len(hq2) > 2:
        heappush(hq1, heappop(hq2))
print(ans)