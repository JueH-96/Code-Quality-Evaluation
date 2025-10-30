import sys
from collections import defaultdict

def read_input():
    N, M = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v, b, c = map(int, sys.stdin.readline().split())
        edges.append((u, v, b, c))
    return N, M, edges

def solve(N, M, edges):
    graph = defaultdict(list)
    for u, v, b, c in edges:
        graph[u].append((v, b, c))

    max_beauty = [0] * (N + 1)
    max_beauty[1] = 1

    for _ in range(N):
        for u in range(1, N + 1):
            for v, b, c in graph[u]:
                if max_beauty[u] * b > max_beauty[v] * c:
                    max_beauty[v] = max_beauty[u] * b / c

    print(max_beauty[N])

def main():
    N, M, edges = read_input()
    solve(N, M, edges)

if __name__ == "__main__":
    main()