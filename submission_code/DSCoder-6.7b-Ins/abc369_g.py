import sys
from collections import defaultdict
from heapq import heappop, heappush

def read_input():
    N = int(sys.stdin.readline())
    graph = defaultdict(list)
    for _ in range(N - 1):
        u, v, w = map(int, sys.stdin.readline().split())
        graph[u].append((v, w))
        graph[v].append((u, w))
    return N, graph

def solve(N, graph):
    for K in range(1, N + 1):
        dist = [float('inf')] * (N + 1)
        dist[1] = 0
        queue = [(0, 1)]
        while queue:
            d, node = heappop(queue)
            if dist[node] < d:
                continue
            for neighbor, w in graph[node]:
                if dist[neighbor] > dist[node] + w:
                    dist[neighbor] = dist[node] + w
                    heappush(queue, (dist[neighbor], neighbor))
        print(max(dist[1:K+1] + dist[K+1:]))

def main():
    N, graph = read_input()
    solve(N, graph)

if __name__ == "__main__":
    main()