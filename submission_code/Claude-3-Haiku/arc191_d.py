from collections import deque

def bfs(graph, s, t):
    queue = deque([(s, 0)])
    visited = set([s])
    while queue:
        node, moves = queue.popleft()
        if node == t:
            return moves
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, moves + 1))
    return -1

def solve():
    n, m, s, t = map(int, input().split())
    graph = [[] for _ in range(n+1)]
    for _ in range(m):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)
    
    # Move piece A to T
    moves_a = bfs(graph, s, t)
    if moves_a == -1:
        print(-1)
        return
    
    # Move piece B to S
    moves_b = bfs(graph, t, s)
    if moves_b == -1:
        print(-1)
        return
    
    print(moves_a + moves_b)

solve()