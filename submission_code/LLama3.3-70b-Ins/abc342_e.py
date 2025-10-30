import sys
import heapq

def read_input():
    N, M = map(int, sys.stdin.readline().split())
    trains = []
    for _ in range(M):
        l, d, k, c, A, B = map(int, sys.stdin.readline().split())
        trains.append((l, d, k, c, A, B))
    return N, trains

def solve(N, trains):
    graph = [[] for _ in range(N + 1)]
    for l, d, k, c, A, B in trains:
        for i in range(k):
            t = l + i * d
            graph[A].append((t, c, B))

    def dijkstra(start):
        dist = [-float('inf')] * (N + 1)
        dist[start] = float('inf')
        pq = [(-float('inf'), start)]
        while pq:
            t, node = heapq.heappop(pq)
            t = -t
            for nt, c, neighbor in graph[node]:
                if nt <= t and t + c > dist[neighbor]:
                    dist[neighbor] = t + c
                    heapq.heappush(pq, (-dist[neighbor], neighbor))
        return dist

    dist = dijkstra(1)
    for i in range(1, N):
        if dist[i] == -float('inf'):
            print("Unreachable")
        else:
            print(dist[i])

N, trains = read_input()
solve(N, trains)