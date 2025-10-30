import sys
from collections import defaultdict, deque

def read_input():
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    edges = [(int(data[i]), int(data[i+1])) for i in range(1, 2*n-1, 2)]
    c = list(map(int, data[2*n-1:]))
    return n, edges, c

def build_tree(edges):
    tree = defaultdict(list)
    for u, v in edges:
        tree[u].append(v)
        tree[v].append(u)
    return tree

def bfs(tree, start):
    queue = deque([start])
    parent = {start: None}
    distance = {start: 0}
    while queue:
        node = queue.popleft()
        for neighbor in tree[node]:
            if neighbor not in parent:
                parent[neighbor] = node
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return distance

def calculate_f(n, c, distances):
    return sum(c[i-1] * distances[i] for i in range(1, n+1))

def find_min_f(n, edges, c):
    tree = build_tree(edges)
    distances = bfs(tree, 1)
    f_values = [calculate_f(n, c, bfs(tree, i+1)) for i in range(n)]
    return min(f_values)

n, edges, c = read_input()
print(find_min_f(n, edges, c))