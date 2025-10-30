import collections

def solve():
    n, m = map(int, input().split())
    edges = []
    graph = collections.defaultdict(list)
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))
        graph[u].append(v)

    q = collections.deque([(1, [1], 0)])
    visited = set()
    min_cycle_len = float('inf')

    while q:
        curr_node, path, dist = q.popleft()
        
        for neighbor in graph[curr_node]:
            if neighbor in path:
                if neighbor == 1:
                    min_cycle_len = min(min_cycle_len, dist + 1)
            else:
                new_path = path + [neighbor]
                q.append((neighbor, new_path, dist + 1))

    if min_cycle_len == float('inf'):
        print(-1)
    else:
        print(min_cycle_len)

solve()