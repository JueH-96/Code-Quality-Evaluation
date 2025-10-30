from collections import defaultdict, deque
import sys

def find_min_cycle_containing_vertex_1(n, m, edges):
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)

    # To find the minimum cycle containing vertex 1, we can use a modified BFS
    # to track the distance from vertex 1 and also check for cycles.
    
    # Distance from vertex 1
    distance = {i: float('inf') for i in range(1, n + 1)}
    distance[1] = 0
    queue = deque([1])
    parent = {1: None}
    
    while queue:
        current = queue.popleft()
        
        for neighbor in graph[current]:
            if distance[neighbor] == float('inf'):
                distance[neighbor] = distance[current] + 1
                parent[neighbor] = current
                queue.append(neighbor)
            elif parent[current] != neighbor:  # Found a back edge indicating a cycle
                # Calculate the cycle length
                cycle_length = distance[current] + distance[neighbor] + 1
                if cycle_length >= 3:  # We need at least 3 edges to form a cycle
                    return cycle_length

    return -1

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    n, m = map(int, data[0].split())
    edges = [tuple(map(int, line.split())) for line in data[1:m + 1]]
    
    result = find_min_cycle_containing_vertex_1(n, m, edges)
    print(result)

if __name__ == "__main__":
    main()