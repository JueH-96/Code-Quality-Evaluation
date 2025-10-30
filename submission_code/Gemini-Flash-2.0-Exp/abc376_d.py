from collections import defaultdict, deque

def solve():
    n, m = map(int, input().split())
    adj = defaultdict(list)
    for _ in range(m):
        a, b = map(int, input().split())
        adj[a].append(b)

    min_cycle_len = float('inf')

    def bfs(start_node):
        nonlocal min_cycle_len
        q = deque([(start_node, [start_node])])
        visited = {start_node}

        while q:
            node, path = q.popleft()

            for neighbor in adj[node]:
                if neighbor == start_node:
                    min_cycle_len = min(min_cycle_len, len(path))
                elif neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, path + [neighbor]))

    bfs(1)

    if min_cycle_len == float('inf'):
        print("-1")
    else:
        print(min_cycle_len)

solve()