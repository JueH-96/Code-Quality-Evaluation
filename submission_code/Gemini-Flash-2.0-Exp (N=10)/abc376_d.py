from collections import defaultdict, deque

def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))

    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)

    min_cycle_len = float('inf')
    
    q = deque([(1, [1])])
    
    while q:
        curr_node, path = q.popleft()
        
        for neighbor in adj[curr_node]:
            if neighbor == 1:
                min_cycle_len = min(min_cycle_len, len(path))
            elif neighbor not in path:
                q.append((neighbor, path + [neighbor]))
    
    if min_cycle_len == float('inf'):
        print("-1")
    else:
        print(min_cycle_len)

solve()