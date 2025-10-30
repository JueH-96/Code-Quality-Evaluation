# YOUR CODE HERE
import sys
from collections import deque

def main():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    K = int(data[index])
    index += 1
    
    # Build the graph
    graph = [[] for _ in range(N + 1)]
    for _ in range(M):
        a = int(data[index])
        index += 1
        b = int(data[index])
        index += 1
        graph[a].append(b)
        graph[b].append(a)
    
    # Guards information
    guards = []
    for _ in range(K):
        p = int(data[index])
        index += 1
        h = int(data[index])
        index += 1
        guards.append((p, h))
    
    # Set to store guarded vertices
    guarded = set()
    
    # Perform BFS for each guard
    for p, h in guards:
        # BFS setup
        queue = deque([(p, 0)])  # (current_vertex, current_distance)
        visited = set()
        visited.add(p)
        
        while queue:
            current_vertex, current_distance = queue.popleft()
            
            # Mark the current vertex as guarded
            guarded.add(current_vertex)
            
            # If we can still move further
            if current_distance < h:
                for neighbor in graph[current_vertex]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, current_distance + 1))
    
    # Prepare the result
    guarded_list = sorted(guarded)
    print(len(guarded_list))
    print(" ".join(map(str, guarded_list)))

if __name__ == "__main__":
    main()