import heapq
import sys
input = sys.stdin.readline

def main():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    edges = [list(map(int, input().split())) for _ in range(M)]
    G = [[] for _ in range(N)]
    for u, v, b in edges:
        u -= 1
        v -= 1
        G[u].append((v, b))
        G[v].append((u, b))
    min_A = min(A)
    min_v = A.index(min_A)
    dist = [float('inf')] * N
    dist[0] = A[0]
    hq = [(A[0], 0)]
    heapq.heapify(hq)
    while hq:
        d, v = heapq.heappop(hq)
        if dist[v] < d:
            continue
        for to, b in G[v]:
            if d + b + A[to] < dist[to]:
                dist[to] = d + b + A[to]
                heapq.heappush(hq, (dist[to], to))
            if v != min_v and d + b + min_A < dist[to]:
                dist[to] = d + b + min_A
                heapq.heappush(hq, (dist[to], to))
    print(*dist[1:], sep='
')

if __name__ == "__main__":
    main()