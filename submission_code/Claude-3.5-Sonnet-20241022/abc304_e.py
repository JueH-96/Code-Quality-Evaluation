from collections import deque

def has_path(graph, start, end, n):
    visited = [False] * (n+1)
    queue = deque([start])
    visited[start] = True
    
    while queue:
        curr = queue.popleft()
        if curr == end:
            return True
            
        for next_node in graph[curr]:
            if not visited[next_node]:
                visited[next_node] = True
                queue.append(next_node)
                
    return False

# Read input
N, M = map(int, input().split())
graph = [[] for _ in range(N+1)]

for _ in range(M):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)

K = int(input())
forbidden = []
for _ in range(K):
    x, y = map(int, input().split())
    forbidden.append((x,y))

Q = int(input())
queries = []
for _ in range(Q):
    p, q = map(int, input().split())
    queries.append((p,q))

# Process each query
for p, q in queries:
    # Add temporary edge
    graph[p].append(q)
    graph[q].append(p)
    
    # Check if graph remains good
    is_good = True
    for x, y in forbidden:
        if has_path(graph, x, y, N):
            is_good = False
            break
    
    print("Yes" if is_good else "No")
    
    # Remove temporary edge
    graph[p].remove(q)
    graph[q].remove(p)