import heapq
N = int(input())
slimes = [tuple(map(int, input().split())) for _ in range(N)]
slimes.sort()
hq = []
for s, c in slimes:
    if hq and hq[0] == s:
        c += 1
    while c > 1:
        c -= 2
        if hq:
            heapq.heappush(hq, 2 * heapq.heappop(hq))
        else:
            heapq.heappush(hq, 2 * s)
    if c:
        heapq.heappush(hq, s)
print(len(hq))