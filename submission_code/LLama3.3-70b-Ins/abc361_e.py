import sys
from collections import defaultdict

def read_ints():
    return list(map(int, sys.stdin.readline().split()))

def main():
    N = int(sys.stdin.readline())
    edges = []
    for _ in range(N - 1):
        A, B, C = read_ints()
        edges.append((A - 1, B - 1, C))

    # Create adjacency list
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    # Prim's algorithm to find MST
    mst = []
    visited = [False] * N
    visited[0] = True
    edges = [(0, 0, 0)]  # (u, v, w)
    while edges:
        edges.sort(key=lambda x: x[2])
        u, v, w = edges.pop(0)
        if not visited[v]:
            visited[v] = True
            mst.append((u, v, w))
            for x, y in graph[v]:
                if not visited[x]:
                    edges.append((v, x, y))

    # Find the minimum travel distance
    total_distance = 0
    for u, v, w in mst:
        total_distance += w

    # Since we don't need to return to the starting city, we can simply return the total distance
    print(total_distance)

if __name__ == "__main__":
    main()