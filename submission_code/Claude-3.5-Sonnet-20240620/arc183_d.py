# YOUR CODE HERE
from collections import defaultdict

def read_input():
    N = int(input())
    graph = defaultdict(list)
    for _ in range(N - 1):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
    return N, graph

def find_leaves(graph):
    return [node for node in graph if len(graph[node]) == 1]

def find_path(graph, start, end):
    queue = [(start, [start])]
    visited = set()
    
    while queue:
        (node, path) = queue.pop(0)
        if node not in visited:
            visited.add(node)
            
            if node == end:
                return path
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    
    return []

def remove_nodes(graph, node1, node2):
    for neighbor in graph[node1]:
        graph[neighbor].remove(node1)
    for neighbor in graph[node2]:
        graph[neighbor].remove(node2)
    del graph[node1]
    del graph[node2]

def solve():
    N, graph = read_input()
    
    for _ in range(N // 2):
        leaves = find_leaves(graph)
        leaf1, leaf2 = leaves[0], leaves[-1]
        
        print(f"{leaf1} {leaf2}")
        
        remove_nodes(graph, leaf1, leaf2)

solve()