# YOUR CODE HERE
import sys
from collections import defaultdict
from heapq import heappush, heappop

def read_input():
    N = int(sys.stdin.readline())
    graph = defaultdict(list)
    for _ in range(N - 1):
        u, v, l = map(int, sys.stdin.readline().split())
        graph[u].append((v, l))
        graph[v].append((u, l))
    return N, graph

def dijkstra(graph, start):
    distances = {start: 0}
    pq = [(0, start)]
    while pq:
        current_dist, current_node = heappop(pq)
        if current_dist > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node]:
            distance = current_dist + weight
            if distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance
                heappush(pq, (distance, neighbor))
    return distances

def dfs(graph, node, parent, distances):
    subtree_max = 0
    for child, weight in graph[node]:
        if child != parent:
            child_max = dfs(graph, child, node, distances)
            subtree_max = max(subtree_max, child_max + weight)
    distances[node] = subtree_max
    return subtree_max

def solve(N, graph):
    root_distances = dijkstra(graph, 1)
    max_distances = {}
    dfs(graph, 1, None, max_distances)
    
    sorted_distances = sorted([(root_distances[i], max_distances[i]) for i in range(1, N+1)], reverse=True)
    
    current_sum = 0
    for k in range(1, N+1):
        if k <= len(sorted_distances):
            current_sum += sorted_distances[k-1][0]
        print(current_sum * 2)

N, graph = read_input()
solve(N, graph)