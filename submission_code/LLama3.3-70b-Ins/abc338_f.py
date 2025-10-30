import sys
from itertools import permutations

def read_input():
    N, M = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        edges.append((u-1, v-1, w))
    return N, edges

def has_path(edges, N, path):
    graph = [[] for _ in range(N)]
    for u, v, _ in edges:
        graph[u].append(v)
    for i in range(len(path) - 1):
        if path[i+1] not in graph[path[i]]:
            return False
    return True

def calculate_weight(edges, path):
    weight = 0
    for i in range(len(path) - 1):
        for u, v, w in edges:
            if u == path[i] and v == path[i+1]:
                weight += w
                break
    return weight

def solve(N, edges):
    min_weight = float('inf')
    for path in permutations(range(N)):
        if has_path(edges, N, path):
            weight = calculate_weight(edges, path)
            min_weight = min(min_weight, weight)
    if min_weight == float('inf'):
        print("No")
    else:
        print(min_weight)

def main():
    N, edges = read_input()
    solve(N, edges)

if __name__ == "__main__":
    main()