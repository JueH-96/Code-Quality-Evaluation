from typing import List, Dict
from collections import defaultdict
from heapq import heappush, heappop

class LargeGraph:
    def __init__(self, N: int):
        self.N = N
        self.edges: Dict[int, List[int]] = defaultdict(list)
        self.hps: Dict[int, List[int]] = dict()  # To store the top 10 largest connected vertices for efficiency

    def add_edge(self, u: int, v: int):
        if v not in self.edges[u]:
            heappush(self.hps[u], -v)  # Max-heap by storing negative values
            self.edges[u].append(v)
        if u not in self.edges[v]:
            heappush(self.hps[v], -u)  # Max-heap by storing negative values
            self.edges[v].append(u)
        self.prune_heap(u)
        self.prune_heap(v)

    def prune_heap(self, u: int):
        # Keep only the top 10 largest vertex numbers in the heap for each vertex
        if len(self.hps[u]) > 10:
            heappop(self.hps[u])

    def kth_largest(self, v: int, k: int) -> int:
        # Adjust k since we want the kth largest but the heap is 0-indexed
        k -= 1
        # Adjust the top values heap if there are fewer than 10 connections
        if len(self.hps[v]) < 10:
            connected = sorted(self.edges[v], reverse=True)
            self.hps[v] = [-i for i in connected]

        # If k is still too large, return -1
        if k >= len(self.hps[v]):
            return -1
        return -self.hps[v][k]

# Read input
N, Q = map(int, input().split())
graph = LargeGraph(N)

# Process queries
for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        graph.add_edge(query[1], query[2])
    elif query[0] == 2:
        print(graph.kth_largest(query[1], query[2]))