from collections import deque

def solve():
    n, m, k = map(int, input().split())
    adj = [[] for _ in range(n)]
    for _ in range(m):
        u, v = map(int, input().split())
        adj[u - 1].append(v - 1)
        adj[v - 1].append(u - 1)

    guards = []
    for _ in range(k):
        p, h = map(int, input().split())
        guards.append((p - 1, h))

    guarded_vertices = set()

    for start_node, stamina in guards:
        q = deque([(start_node, 0)])
        visited = [-1] * n
        visited[start_node] = 0

        while q:
            current_node, dist = q.popleft()

            if dist <= stamina:
                guarded_vertices.add(current_node + 1)

                for neighbor in adj[current_node]:
                    if visited[neighbor] == -1:
                        visited[neighbor] = dist + 1
                        q.append((neighbor, dist + 1))

    sorted_guarded = sorted(list(guarded_vertices))
    print(len(sorted_guarded))
    print(*sorted_guarded)

solve()