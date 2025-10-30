import sys
from collections import defaultdict, deque

def read_input():
    n = int(sys.stdin.readline().strip())
    edges = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(n-1)]
    return n, edges

def solve():
    n, edges = read_input()
    graph = defaultdict(list)
    degrees = [0] * (n+1)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
        degrees[u] += 1
        degrees[v] += 1
    leaves = deque([i for i in range(1, n+1) if degrees[i] == 1])
    levels = [0] * (n+1)
    while leaves:
        new_leaves = []
        for u in leaves:
            for v in graph[u]:
                if degrees[v] > 0:
                    degrees[v] -= 1
                    levels[v] = levels[u] + 1
                    if degrees[v] == 1:
                        new_leaves.append(v)
        leaves = deque(new_leaves)
    return sorted(levels[1:])

print(' '.join(map(str, solve())))