import sys
import heapq

def solve():
    N, M = map(int, sys.stdin.readline().split())
    trains = [[] for _ in range(N)]
    for _ in range(M):
        l, d, k, c, A, B = map(int, sys.stdin.readline().split())
        for t in range(l, l+d*k, d):
            trains[A-1].append((t, c, A-1, B-1))
    for i in range(N-1):
        heap = []
        for t, c, A, B in trains[i]:
            heapq.heappush(heap, (t, c, A, B))
        while heap:
            t, c, A, B = heapq.heappop(heap)
            if B == N-1:
                print(t+c)
                break
            for tt, cc, AA, BB in trains[B]:
                if tt >= t+c:
                    heapq.heappush(heap, (tt, cc, AA, BB))
        else:
            print('Unreachable')

solve()