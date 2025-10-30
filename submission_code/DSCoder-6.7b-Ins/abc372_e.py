import sys
import heapq

class Graph:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n+1)]

    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)

    def kth_largest(self, v, k):
        # Use a max heap to store the k largest vertices
        heap = []
        for neighbor in self.adj[v]:
            heapq.heappush(heap, -neighbor)
            if len(heap) > k:
                heapq.heappop(heap)
        if len(heap) < k:
            return -1
        else:
            return -heap[0]

def main():
    n, q = map(int, sys.stdin.readline().split())
    graph = Graph(n)
    for _ in range(q):
        query = list(map(int, sys.stdin.readline().split()))
        if query[0] == 1:
            graph.add_edge(query[1], query[2])
        else:
            print(graph.kth_largest(query[1], query[2]))

if __name__ == "__main__":
    main()