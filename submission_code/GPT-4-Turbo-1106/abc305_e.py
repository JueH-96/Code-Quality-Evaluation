from collections import deque

def bfs(graph, start, stamina):
    visited = set()
    queue = deque([(start, stamina)])
    while queue:
        current, stam = queue.popleft()
        if current not in visited:
            visited.add(current)
            if stam > 0:
                for neighbor in graph[current]:
                    queue.append((neighbor, stam - 1))
    return visited

def main():
    N, M, K = map(int, input().split())
    graph = {i: [] for i in range(1, N + 1)}
    
    for _ in range(M):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
    
    guarded = set()
    for _ in range(K):
        p, h = map(int, input().split())
        guarded |= bfs(graph, p, h)
    
    guarded = sorted(guarded)
    print(len(guarded))
    print(' '.join(map(str, guarded)))

if __name__ == "__main__":
    main()