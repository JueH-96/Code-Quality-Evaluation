import heapq

N = int(input())
TD = [list(map(int, input().split())) for _ in range(N)]

TD.sort(key=lambda x: x[1])
hq = []
ans = 0
for t, d in TD:
    if hq and -hq[0] < t:
        heapq.heappop(hq)
    heapq.heappush(hq, -d)
    ans = max(ans, len(hq))

print(ans)