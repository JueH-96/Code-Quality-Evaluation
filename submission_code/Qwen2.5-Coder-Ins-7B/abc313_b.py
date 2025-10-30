# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

# Initialize the graph
graph = {i: [] for i in range(1, N + 1)}

# Read the superiority relations
for i in range(M):
    A = int(data[2 + 2 * i])
    B = int(data[3 + 2 * i])
    graph[A].append(B)

# Function to perform DFS and check for cycles
def dfs(node, visited, stack):
    visited[node] = True
    stack[node] = True
    
    for neighbor in graph[node]:
        if not visited[neighbor]:
            if dfs(neighbor, visited, stack):
                return True
        elif stack[neighbor]:
            return True
    
    stack[node] = False
    return False

# Check for cycles in the graph
visited = [False] * (N + 1)
stack = [False] * (N + 1)

for i in range(1, N + 1):
    if not visited[i]:
        if dfs(i, visited, stack):
            print(-1)
            sys.exit(0)

# If no cycles, determine the strongest programmer
in_degree = [0] * (N + 1)

for node in graph:
    for neighbor in graph[node]:
        in_degree[neighbor] += 1

strongest = []
for i in range(1, N + 1):
    if in_degree[i] == 0:
        strongest.append(i)

if len(strongest) == 1:
    print(strongest[0])
else:
    print(-1)