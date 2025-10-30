import sys
from collections import defaultdict

def dfs(node, parent, depth, max_depth, max_node, distances):
    distances[node] = depth
    if depth > max_depth[0]:
        max_depth[0] = depth
        max_node[0] = node
    for child, length in graph[node]:
        if child != parent:
            dfs(child, node, depth + length, max_depth, max_node, distances)

def find_diameter():
    max_depth = [0]
    max_node = [1]
    distances = defaultdict(int)
    dfs(1, -1, 0, max_depth, max_node, distances)
    start = max_node[0]
    max_depth = [0]
    max_node = [start]
    distances = defaultdict(int)
    dfs(start, -1, 0, max_depth, max_node, distances)
    end = max_node[0]
    return max_depth[0]

def solve():
    N = int(input())
    graph = defaultdict(list)
    for _ in range(N - 1):
        U, V, L = map(int, input().split())
        graph[U].append((V, L))
        graph[V].append((U, L))
    
    diameter = find_diameter()
    result = [0] * N
    result[0] = 2 * (N - 1)
    for i in range(1, N):
        result[i] = result[i - 1] + 2 * (diameter // (i + 1))
    
    for i in range(1, N):
        result[i] = max(result[i], result[i - 1])
    
    for res in result:
        print(res)

if __name__ == "__main__":
    solve()