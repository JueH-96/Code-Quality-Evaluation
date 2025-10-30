import sys
import heapq

def max_beauty_cost_ratio(N, M, edges):
    graph = [[] for _ in range(N + 1)]
    for u, v, b, c in edges:
        graph[u].append((v, b, c))

    # Dijkstra-like algorithm to find the maximum beauty/cost ratio path
    max_ratio = -1
    priority_queue = [(-float('inf'), 0, 1)]  # (current_ratio, current_cost, current_node)
    visited = set()

    while priority_queue:
        current_ratio, current_cost, current_node = heapq.heappop(priority_queue)
        current_ratio = -current_ratio

        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == N:
            max_ratio = max(max_ratio, current_ratio)
            continue

        for neighbor, beauty, cost in graph[current_node]:
            new_cost = current_cost + cost
            new_beauty = current_ratio * new_cost + beauty
            new_ratio = new_beauty / new_cost
            heapq.heappush(priority_queue, (-new_ratio, new_cost, neighbor))

    return max_ratio

def main():
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
        index += 4
        edges.append((u, v, b, c))

    result = max_beauty_cost_ratio(N, M, edges)
    print(f"{result:.12f}")

if __name__ == "__main__":
    main()