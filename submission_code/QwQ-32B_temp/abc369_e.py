import sys
import heapq
from itertools import permutations

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1

    edges = []
    adj = [[] for _ in range(N + 1)]  # 1-based indexing
    for _ in range(M):
        u = int(input[ptr]); ptr += 1
        v = int(input[ptr]); ptr += 1
        t = int(input[ptr]); ptr += 1
        edges.append((u, v, t))
        adj[u].append((v, t))
        adj[v].append((u, t))

    # Precompute All-Pairs Shortest Paths using Dijkstra's algorithm
    INF = float('inf')
    apsp = [[INF] * (N + 1) for _ in range(N + 1)]

    for start in range(1, N + 1):
        dist = [INF] * (N + 1)
        dist[start] = 0
        heap = []
        heapq.heappush(heap, (0, start))
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                if dist[v] > d + w:
                    dist[v] = d + w
                    heapq.heappush(heap, (dist[v], v))
        for v in range(1, N + 1):
            apsp[start][v] = dist[v]

    Q = int(input[ptr]); ptr += 1
    for _ in range(Q):
        K = int(input[ptr]); ptr += 1
        bridges = list(map(int, input[ptr:ptr + K]))
        ptr += K
        query_edges = [edges[b - 1] for b in bridges]
        Kq = len(query_edges)
        min_total = INF

        for perm in permutations(query_edges):
            for mask in range(0, 1 << Kq):
                cost = 0
                prev_end = None
                for i in range(Kq):
                    u, v, t = perm[i]
                    bit = (mask >> i) & 1
                    if bit:
                        s, e = v, u
                    else:
                        s, e = u, v
                    if i == 0:
                        cost += apsp[1][s]
                    else:
                        cost += apsp[prev_end][s]
                    cost += t
                    prev_end = e
                cost += apsp[prev_end][N]
                if cost < min_total:
                    min_total = cost
        print(min_total)

if __name__ == "__main__":
    main()