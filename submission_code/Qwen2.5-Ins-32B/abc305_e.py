import sys
from collections import deque

def bfs(graph, start, stamina):
    visited = [False] * (N + 1)
    visited[start] = True
    queue = deque([(start, 0)])
    guarded = set()
    
    while queue:
        vertex, dist = queue.popleft()
        if dist <= stamina:
            guarded.add(vertex)
            for neighbor in graph[vertex]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, dist + 1))
        else:
            break
    return guarded

def main():
    input = sys.stdin.read
    data = input().split()
    
    N, M, K = map(int, data[:3])
    data = data[3:]
    
    graph = [[] for _ in range(N + 1)]
    for i in range(M):
        a, b = map(int, (data[i * 2], data[i * 2 + 1]))
        graph[a].append(b)
        graph[b].append(a)
    
    data = data[2 * M:]
    guards = [(int(data[i * 2]), int(data[i * 2 + 1])) for i in range(K)]
    
    guarded_vertices = set()
    for p, h in guards:
        guarded_vertices.update(bfs(graph, p, h))
    
    guarded_vertices = sorted(guarded_vertices)
    print(len(guarded_vertices))
    print(' '.join(map(str, guarded_vertices)))

if __name__ == "__main__":
    main()