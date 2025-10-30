# YOUR CODE HERE
import sys
from collections import deque

def bfs(graph, start, max_distance):
    visited = [False] * (len(graph) + 1)
    queue = deque([(start, 0)])
    visited[start] = True
    guarded = set()
    
    while queue:
        current, distance = queue.popleft()
        if distance <= max_distance:
            guarded.add(current)
            for neighbor in graph[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, distance + 1))
    
    return guarded

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    
    N = int(input[index])
    M = int(input[index + 1])
    K = int(input[index + 2])
    index += 3
    
    graph = [[] for _ in range(N + 1)]
    for _ in range(M):
        a = int(input[index])
        b = int(input[index + 1])
        graph[a].append(b)
        graph[b].append(a)
        index += 2
    
    guards = []
    for _ in range(K):
        p = int(input[index])
        h = int(input[index + 1])
        guards.append((p, h))
        index += 2
    
    guarded_vertices = set()
    for p, h in guards:
        guarded_vertices.update(bfs(graph, p, h))
    
    guarded_vertices = sorted(guarded_vertices)
    print(len(guarded_vertices))
    print(' '.join(map(str, guarded_vertices)))

if __name__ == "__main__":
    main()