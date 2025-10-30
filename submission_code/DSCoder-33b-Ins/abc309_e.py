import sys
from collections import defaultdict

def dfs(graph, node, depth, y):
    if depth > y:
        return set()
    covered = {node}
    for child in graph[node]:
        covered |= dfs(graph, child, depth + 1, y)
    return covered

def solve():
    n, m = map(int, sys.stdin.readline().split())
    parents = list(map(int, sys.stdin.readline().split()))
    graph = defaultdict(list)
    for i in range(2, n + 1):
        graph[parents[i - 2]].append(i)
    covered = set()
    for _ in range(m):
        x, y = map(int, sys.stdin.readline().split())
        covered |= dfs(graph, x, 1, y)
    print(len(covered))

solve()