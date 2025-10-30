import sys
from collections import defaultdict

def dfs(graph, node, visited):
    visited.add(node)
    total_distance = 0
    for neighbor, distance in graph[node]:
        if neighbor not in visited:
            total_distance += distance + dfs(graph, neighbor, visited)
    return total_distance

def main():
    N = int(sys.stdin.readline())
    graph = defaultdict(list)
    for _ in range(N - 1):
        A, B, C = map(int, sys.stdin.readline().split())
        graph[A].append((B, C))
        graph[B].append((A, C))

    visited = set()
    min_distance = dfs(graph, 1, visited)
    print(min_distance)

if __name__ == "__main__":
    main()