from collections import defaultdict, deque

def solve():
    N, M, K = map(int, input().split())
    graph = defaultdict(list)
    guards = defaultdict(int)
    for _ in range(M):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
    for _ in range(K):
        p, h = map(int, input().split())
        guards[p] = h
    guarded = set()
    for guard, stamina in guards.items():
        visited = set()
        queue = deque([(guard, 0)])
        while queue:
            vertex, distance = queue.popleft()
            if distance > stamina:
                break
            visited.add(vertex)
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
        guarded.update(visited)
    guarded = sorted(list(guarded))
    print(len(guarded))
    print(' '.join(map(str, guarded)))

solve()