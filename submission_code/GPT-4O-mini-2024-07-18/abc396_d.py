from collections import defaultdict, deque
import sys

def min_xor_path(n, edges):
    graph = defaultdict(list)
    
    # Build the graph
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    # BFS to find all paths from 1 to N
    min_xor = float('inf')
    queue = deque([(1, 0, {1})])  # (current_node, current_xor, visited_set)
    
    while queue:
        current_node, current_xor, visited = queue.popleft()
        
        if current_node == n:
            min_xor = min(min_xor, current_xor)
            continue
        
        for neighbor, weight in graph[current_node]:
            if neighbor not in visited:
                new_xor = current_xor ^ weight
                new_visited = visited | {neighbor}
                queue.append((neighbor, new_xor, new_visited))
    
    return min_xor

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    n, m = map(int, data[0].split())
    edges = []
    
    for i in range(1, m + 1):
        u, v, w = map(int, data[i].split())
        edges.append((u, v, w))
    
    result = min_xor_path(n, edges)
    print(result)

if __name__ == "__main__":
    main()