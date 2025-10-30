from sys import stdin, stdout
from collections import deque

def read_ints():
    return map(int, stdin.readline().strip().split())

def bfs(graph, start, n):
    distances = [-1] * (n + 1)
    queue = deque([start])
    distances[start] = 0
    while queue:
        node = queue.popleft()
        current_distance = distances[node]
        for neighbor in graph[node]:
            if distances[neighbor] == -1:
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)
    return distances

def main():
    N, M, S, T = read_ints()
    graph = [[] for _ in range(N + 1)]
    
    for _ in range(M):
        u, v = read_ints()
        graph[u].append(v)
        graph[v].append(u)
    
    # BFS from S and T to get distances to all nodes
    dist_from_S = bfs(graph, S, N)
    dist_from_T = bfs(graph, T, N)
    
    # If it's impossible to swap, the distance from S to T must be infinite (or disconnected)
    if dist_from_S[T] == -1:
        stdout.write("-1
")
        return
    
    # We need to find the minimum number of moves to swap positions
    # This is equivalent to finding the shortest path where A reaches T and B reaches S without colliding
    min_moves = float('inf')
    
    # We consider each vertex as a potential meeting point
    for i in range(1, N + 1):
        if dist_from_S[i] != -1 and dist_from_T[i] != -1:
            # Calculate the total moves if A and B meet at vertex i and then swap
            total_moves = dist_from_S[i] + dist_from_T[i]
            if total_moves < min_moves:
                min_moves = total_moves
    
    if min_moves == float('inf'):
        stdout.write("-1
")
    else:
        stdout.write(f"{min_moves}
")

if __name__ == "__main__":
    main()