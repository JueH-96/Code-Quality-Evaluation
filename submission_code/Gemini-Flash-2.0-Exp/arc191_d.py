from collections import deque

def solve():
    n, m, s, t = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))

    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def bfs(start_a, start_b, target_a, target_b):
        q = deque([(start_a, start_b, 0)])
        visited = set()
        visited.add((start_a, start_b))

        while q:
            curr_a, curr_b, dist = q.popleft()

            if curr_a == target_a and curr_b == target_b:
                return dist

            # Move A
            for next_a in adj[curr_a]:
                if next_a != curr_b and (next_a, curr_b) not in visited:
                    q.append((next_a, curr_b, dist + 1))
                    visited.add((next_a, curr_b))

            # Move B
            for next_b in adj[curr_b]:
                if next_b != curr_a and (curr_a, next_b) not in visited:
                    q.append((curr_a, next_b, dist + 1))
                    visited.add((curr_a, next_b))
        
        return -1

    result = bfs(s, t, t, s)
    print(result)

solve()