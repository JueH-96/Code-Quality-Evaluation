import sys
from collections import defaultdict, deque

def find_stars(n, edges):
    graph = defaultdict(list)
    degree = [0] * (n + 1)
    
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
        degree[u] += 1
        degree[v] += 1
    
    leaves = deque([i for i in range(1, n + 1) if degree[i] == 1])
    levels = []
    
    while leaves:
        level_size = len(leaves)
        if level_size > 1:
            levels.append(level_size)
        
        for _ in range(level_size):
            leaf = leaves.popleft()
            for neighbor in graph[leaf]:
                degree[neighbor] -= 1
                if degree[neighbor] == 1:
                    leaves.append(neighbor)
    
    levels.sort()
    print(*levels)

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    edges = [(int(data[i]), int(data[i + 1])) for i in range(1, 2 * (n - 1), 2)]
    find_stars(n, edges)