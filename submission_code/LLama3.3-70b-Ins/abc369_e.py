import sys
import heapq

def read_input():
    N, M = map(int, sys.stdin.readline().split())
    bridges = []
    for _ in range(M):
        u, v, t = map(int, sys.stdin.readline().split())
        bridges.append((u, v, t))
    Q = int(sys.stdin.readline())
    queries = []
    for _ in range(Q):
        K = int(sys.stdin.readline())
        B = list(map(int, sys.stdin.readline().split()))
        queries.append(B)
    return N, M, bridges, Q, queries

def dijkstra(N, bridges, start, end, must_use):
    dist = [float('inf')] * (N + 1)
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        curr_dist, curr_node = heapq.heappop(pq)
        if curr_node == end:
            return curr_dist
        for u, v, t in bridges:
            if u == curr_node:
                if (u, v) in must_use or (v, u) in must_use:
                    must_use.remove((u, v))
                    must_use.remove((v, u))
                if dist[v] > curr_dist + t:
                    dist[v] = curr_dist + t
                    heapq.heappush(pq, (dist[v], v))
            elif v == curr_node:
                if (u, v) in must_use or (v, u) in must_use:
                    must_use.remove((u, v))
                    must_use.remove((v, u))
                if dist[u] > curr_dist + t:
                    dist[u] = curr_dist + t
                    heapq.heappush(pq, (dist[u], u))
    return float('inf')

def solve(N, M, bridges, Q, queries):
    for B in queries:
        must_use = set()
        for b in B:
            u, v, t = bridges[b - 1]
            must_use.add((u, v))
            must_use.add((v, u))
        print(dijkstra(N, bridges, 1, N, must_use))

def main():
    N, M, bridges, Q, queries = read_input()
    solve(N, M, bridges, Q, queries)

if __name__ == '__main__':
    main()