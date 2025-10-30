from collections import deque
import sys

def bfs(graph, start, end):
    """Perform BFS to check if there is a path between two nodes."""
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        if node == end:
            return True
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
    return False

def main():
    """Read input and solve the problem."""
    input = sys.stdin.readline
    N, M = map(int, input().split())
    graph = [[] for _ in range(N + 1)]
    
    # Build the graph
    for _ in range(M):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)
    
    K = int(input())
    forbidden_pairs = []
    for _ in range(K):
        x, y = map(int, input().split())
        forbidden_pairs.append((x, y))
    
    Q = int(input())
    for _ in range(Q):
        p, q = map(int, input().split())
        graph[p].append(q)
        graph[q].append(p)
        
        is_good = True
        for x, y in forbidden_pairs:
            if bfs(graph, x, y):
                is_good = False
                break
        
        if is_good:
            print("Yes")
        else:
            print("No")
        
        # Remove the added edge
        graph[p].remove(q)
        graph[q].remove(p)

if __name__ == "__main__":
    main()