import sys
from collections import Counter, defaultdict, deque
from math import *
from heapq import *
from itertools import accumulate
from bisect import bisect_left, bisect_right

input = lambda: sys.stdin.readline().rstrip()
ii = lambda: int(input())
il = lambda: list(map(int, input().split()))
fl = lambda: list(map(float, input().split()))
iln = lambda n: [ii() for _ in range(n)]
iln2 = lambda n: [il() for _ in range(n)]
fln = lambda n: [fl() for _ in range(n)]
iln3 = lambda n: [[[int(num) for num in input().split()] for _ in range(n)] for _ in range(n)]
MOD = int(1e9 + 7)

# Graph representation and related functions
Graph = list[list[tuple[int, int]]]

def create_graph(n: int) -> Graph:
    graph = [[] for _ in range(n + 1)]
    return graph

def add_undirected_edge(graph: Graph, u: int, v: int):
    graph[u].append(v)
    graph[v].append(u)

def dfs(graph: Graph, start: int, n: int, c: list) -> tuple[int, int]:
    count = [0] * (n + 1)
    edge_costs = [0] * (n + 1)
    visited = [False] * (n + 1)
    stack = [start]
    visited[start] = True
    edge_costs[0] = 0  # Dummy entry for 0-based indexing convenience

    while stack:
        vertex = stack.pop()
        for neighbor in graph[vertex]:
            if not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)
                count[neighbor] = count[vertex] + 1
                edge_costs[neighbor] = edge_costs[vertex] + c[neighbor - 1] * count[vertex]

    return min(edge_costs)

def main():
    n = ii()
    graph = create_graph(n)
    for _ in range(n - 1):
        a, b = il()
        add_undirected_edge(graph, a, b)
    c = il()
    print(dfs(graph, 1, n, c))

if __name__ == "__main__":
    main()