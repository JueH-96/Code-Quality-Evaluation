from heapq import heappop, heappush

def solve():
    N, M = map(int, input().split())
    sets = [set(map(int, input().split()[1:])) for _ in range(N)]
    heap = [(len(s), s) for s in sets]
    heapify(heap)
    visited = set()
    count = 0

    while heap:
        _, s = heappop(heap)
        if 1 in s and M in s:
            print(count)
            return
        for _ in range(len(heap)):
            _, t = heappop(heap)
            if s & t:
                u = s | t
                if u not in visited:
                    visited.add(u)
                    heappush(heap, (len(u), u))
                break
        else:
            print(-1)
            return
        count += 1

    print(-1)

solve()