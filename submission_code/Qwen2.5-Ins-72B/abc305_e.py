from collections import deque, defaultdict

def bfs(graph, start, stamina, n):
    queue = deque([(start, 0)])
    visited = [False] * (n + 1)
    guarded = set()
    
    while queue:
        node, dist = queue.popleft()
        if not visited[node] and dist <= stamina:
            guarded.add(node)
            visited[node] = True
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    queue.append((neighbor, dist + 1))
    
    return guarded

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    
    edges = []
    for i in range(M):
        a = int(data[3 + 2 * i])
        b = int(data[4 + 2 * i])
        edges.append((a, b))
    
    guards = []
    for i in range(K):
        p = int(data[3 + 2 * M + 2 * i])
        h = int(data[4 + 2 * M + 2 * i])
        guards.append((p, h))
    
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    
    guarded_vertices = set()
    for p, h in guards:
        guarded_vertices.update(bfs(graph, p, h, N))
    
    guarded_vertices = sorted(guarded_vertices)
    print(len(guarded_vertices))
    print(" ".join(map(str, guarded_vertices)))

if __name__ == "__main__":
    main()