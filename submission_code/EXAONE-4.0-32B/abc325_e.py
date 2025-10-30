import heapq
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    A = int(next(it))
    B = int(next(it))
    C = int(next(it))
    D = []
    for _ in range(n):
        row = list(map(int, [next(it) for _ in range(n)]))
        D.append(row)
    
    INF = 10**18
    dist0 = [INF] * n
    dist1 = [INF] * n
    dist0[0] = 0
    heap = [(0, 0, 0)]
    
    while heap:
        d, u, s = heapq.heappop(heap)
        if s == 0 and d != dist0[u]:
            continue
        if s == 1 and d != dist1[u]:
            continue
            
        if s == 0:
            for v in range(n):
                if u == v:
                    continue
                nd = d + D[u][v] * A
                if nd < dist0[v]:
                    dist0[v] = nd
                    heapq.heappush(heap, (nd, v, 0))
            if d < dist1[u]:
                dist1[u] = d
                heapq.heappush(heap, (d, u, 1))
        else:
            for v in range(n):
                if u == v:
                    continue
                nd = d + D[u][v] * B + C
                if nd < dist1[v]:
                    dist1[v] = nd
                    heapq.heappush(heap, (nd, v, 1))
                    
    ans = min(dist0[n-1], dist1[n-1])
    print(ans)

if __name__ == '__main__':
    main()