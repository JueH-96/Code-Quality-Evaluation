import sys
import heapq
from collections import defaultdict, deque

input = sys.stdin.read

def solve():
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    edges = []
    graph = defaultdict(list)

    for _ in range(N - 1):
        U = int(data[index])
        V = int(data[index + 1])
        L = int(data[index + 2])
        index += 3
        edges.append((U, V, L))
        graph[U].append((V, L))
        graph[V].append((U, L))

    # Calculate distances from vertex 1 to all other vertices
    distances = dijkstra(graph, 1)

    # Sort edges by their lengths in descending order
    edges.sort(key=lambda x: -x[2])

    # Initialize a union-find structure
    uf = UnionFind(N + 1)

    # Initialize an array to store the maximum distance for each component size
    max_dist = [0] * (N + 1)

    # Process edges in descending order of their lengths
    for u, v, l in edges:
        uf.union(u, v)
        size = uf.size(u)
        max_dist[size] = max(max_dist[size], l)

    # Calculate the results for each K
    results = []
    for K in range(1, N + 1):
        max_length = max_dist[K]
        result = 2 * (distances[1] + max_length)
        results.append(result)

    # Print the results
    for result in results:
        print(result)

def dijkstra(graph, start):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
                self.size[rootX] += self.size[rootY]
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
                self.size[rootY] += self.size[rootX]
            else:
                self.parent[rootY] = rootX
                self.size[rootX] += self.size[rootY]
                self.rank[rootX] += 1

    def size(self, x):
        root = self.find(x)
        return self.size[root]

if __name__ == "__main__":
    solve()