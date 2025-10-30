from collections import defaultdict, deque

def is_bipartite(graph, n):
    color = [-1] * (n + 1)  # -1 means uncolored, 0 and 1 are the two colors
    
    for start in range(1, n + 1):
        if color[start] == -1:  # Not visited yet
            # BFS to check if this component is bipartite
            queue = deque([start])
            color[start] = 0
            
            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    if color[neighbor] == -1:
                        color[neighbor] = 1 - color[node]
                        queue.append(neighbor)
                    elif color[neighbor] == color[node]:
                        return False
    
    return True

# Read input
n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Check for self-loops first
for i in range(m):
    if a[i] == b[i]:
        print("No")
        exit()

# Build graph
graph = defaultdict(list)
for i in range(m):
    graph[a[i]].append(b[i])
    graph[b[i]].append(a[i])

# Check if bipartite
if is_bipartite(graph, n):
    print("Yes")
else:
    print("No")