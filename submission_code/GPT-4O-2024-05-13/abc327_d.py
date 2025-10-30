# YOUR CODE HERE
def is_good_pair(N, M, A, B):
    from collections import defaultdict, deque
    
    # Create adjacency list for the graph
    adj = defaultdict(list)
    for i in range(M):
        adj[A[i]].append(B[i])
        adj[B[i]].append(A[i])
    
    # To store the color of each node, -1 means uncolored
    color = [-1] * (N + 1)
    
    # Function to perform BFS and color the graph
    def bfs(start):
        queue = deque([start])
        color[start] = 0  # Start coloring with 0
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if color[neighbor] == -1:  # If not colored
                    color[neighbor] = 1 - color[node]  # Alternate color
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:  # If same color as parent
                    return False
        return True
    
    # Check all components of the graph
    for i in range(1, N + 1):
        if color[i] == -1:  # If not colored
            if not bfs(i):
                return "No"
    
    return "Yes"

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:2+M]))
B = list(map(int, data[2+M:]))

# Output result
print(is_good_pair(N, M, A, B))