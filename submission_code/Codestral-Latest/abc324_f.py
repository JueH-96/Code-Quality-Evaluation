import heapq
from collections import defaultdict, deque

def max_beauty_per_cost(N, M, edges):
    graph = defaultdict(list)
    for u, v, b, c in edges:
        graph[u].append((v, b, c))

    # Dijkstra's algorithm to find the maximum beauty per cost path
    max_heap = [(-float('inf'), 0, 1)]  # (max_beauty_per_cost, total_cost, current_node)
    visited = set()

    while max_heap:
        current_beauty_per_cost, current_cost, node = heapq.heappop(max_heap)

        if node in visited:
            continue
        visited.add(node)

        if node == N:
            return -current_beauty_per_cost

        for v, b, c in graph[node]:
            if v not in visited:
                new_beauty_per_cost = (current_beauty_per_cost * current_cost + b) / (current_cost + c)
                heapq.heappush(max_heap, (-new_beauty_per_cost, current_cost + c, v))

    return 0

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    M = int(data[index + 1])
    index += 2

    edges = []
    for _ in range(M):
        u = int(data[index])
        v = int(data[index + 1])
        b = int(data[index + 2])
        c = int(data[index + 3])
        edges.append((u, v, b, c))
        index += 4

    result = max_beauty_per_cost(N, M, edges)
    print(f"{result:.15f}")

if __name__ == "__main__":
    main()