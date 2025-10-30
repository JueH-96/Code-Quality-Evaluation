# YOUR CODE HERE
import sys
from collections import deque

def bfs(graph, start):
    visited = [False] * len(graph)
    queue = deque([(start, 0)])
    visited[start] = True
    distances = [-1] * len(graph)
    distances[start] = 0
    
    while queue:
        node, dist = queue.popleft()
        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append((neighbor, dist + 1))
                distances[neighbor] = dist + 1
                
    return distances

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    N1 = int(input[index])
    N2 = int(input[index + 1])
    M = int(input[index + 2])
    index += 3
    
    graph = [[] for _ in range(N1 + N2 + 1)]
    
    for _ in range(M):
        a = int(input[index])
        b = int(input[index + 1])
        graph[a].append(b)
        graph[b].append(a)
        index += 2
    
    # BFS from 1 to find distances to all nodes in the first group
    distances_from_1 = bfs(graph, 1)
    
    # BFS from N1 + N2 to find distances to all nodes in the second group
    distances_from_N1_N2 = bfs(graph, N1 + N2)
    
    max_d = 0
    
    for u in range(1, N1 + 1):
        for v in range(N1 + 1, N1 + N2 + 1):
            if distances_from_1[u] != -1 and distances_from_N1_N2[v] != -1:
                d = distances_from_1[u] + distances_from_N1_N2[v] + 1
                max_d = max(max_d, d)
    
    print(max_d)

if __name__ == "__main__":
    main()