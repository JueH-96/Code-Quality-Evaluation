from collections import deque

def is_bipartite(graph, N):
    color = [-1] * (N + 1)  # -1 for uncolored, 0 and 1 for two colors
    
    for start in range(1, N + 1):
        if color[start] == -1:
            color[start] = 0
            queue = deque([start])
            
            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    if color[neighbor] == -1:
                        color[neighbor] = 1 - color[node]
                        queue.append(neighbor)
                    elif color[neighbor] == color[node]:
                        return False
    
    return True

def is_good_pair(N, A, B):
    # Create an adjacency list representation of the graph
    graph = [[] for _ in range(N + 1)]
    
    for a_i, b_i in zip(A, B):
        if a_i == b_i:
            # If a_i and b_i are the same, we can't satisfy the constraint
            return False
        graph[a_i].append(b_i)
        graph[b_i].append(a_i)
    
    # Check if the graph is bipartite
    return is_bipartite(graph, N)

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Check if (A, B) is a good pair of sequences
if is_good_pair(N, A, B):
    print("Yes")
else:
    print("No")