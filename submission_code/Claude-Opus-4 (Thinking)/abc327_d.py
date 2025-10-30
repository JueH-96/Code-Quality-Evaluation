from collections import deque

def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    # Check if any A_i == B_i
    for i in range(m):
        if a[i] == b[i]:
            return "No"
    
    # Build adjacency list
    graph = [[] for _ in range(n + 1)]
    for i in range(m):
        graph[a[i]].append(b[i])
        graph[b[i]].append(a[i])
    
    # Check if bipartite using BFS
    color = [-1] * (n + 1)  # -1 means uncolored, 0 and 1 are the two colors
    
    for start in range(1, n + 1):
        if color[start] == -1:
            # BFS from this node
            queue = deque([start])
            color[start] = 0
            
            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    if color[neighbor] == -1:
                        color[neighbor] = 1 - color[node]
                        queue.append(neighbor)
                    elif color[neighbor] == color[node]:
                        return "No"
    
    return "Yes"

print(solve())