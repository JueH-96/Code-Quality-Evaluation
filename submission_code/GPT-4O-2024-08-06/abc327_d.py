def is_bipartite_graph(N, M, A, B):
    from collections import deque, defaultdict
    
    # Create adjacency list
    graph = defaultdict(list)
    for i in range(M):
        graph[A[i]].append(B[i])
        graph[B[i]].append(A[i])
    
    # Color array, 0 means uncolored, 1 and -1 are the two colors
    color = [0] * (N + 1)
    
    # Function to check if the graph is bipartite starting from node `start`
    def bfs_check(start):
        queue = deque([start])
        color[start] = 1  # Start coloring with 1
        
        while queue:
            node = queue.popleft()
            current_color = color[node]
            
            for neighbor in graph[node]:
                if color[neighbor] == 0:  # If not colored
                    color[neighbor] = -current_color  # Color with opposite color
                    queue.append(neighbor)
                elif color[neighbor] == current_color:
                    return False  # Conflict found
        
        return True
    
    # Check each component of the graph
    for node in range(1, N + 1):
        if color[node] == 0:  # If not yet colored
            if not bfs_check(node):
                return "No"
    
    return "Yes"

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:2+M]))
B = list(map(int, data[2+M:2+2*M]))

# Solve the problem
result = is_bipartite_graph(N, M, A, B)
print(result)