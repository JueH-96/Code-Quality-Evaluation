import collections

def solve():
    n1, n2, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    adj1 = collections.defaultdict(list)
    adj2 = collections.defaultdict(list)
    
    for u, v in edges:
        if 1 <= u <= n1 and 1 <= v <= n1:
            adj1[u].append(v)
            adj1[v].append(u)
        elif n1 + 1 <= u <= n1 + n2 and n1 + 1 <= v <= n1 + n2:
            adj2[u].append(v)
            adj2[v].append(u)
            
    dist1 = [-1] * (n1 + 1)
    q1 = collections.deque([1])
    dist1[1] = 0
    max_dist_v1 = 0
    
    while q1:
        u = q1.popleft()
        max_dist_v1 = max(max_dist_v1, dist1[u])
        for v in adj1[u]:
            if 1 <= v <= n1 and dist1[v] == -1:
                dist1[v] = dist1[u] + 1
                q1.append(v)
                
    dist2 = [-1] * (n1 + n2 + 1)
    q2 = collections.deque([n1 + n2])
    dist2[n1 + n2] = 0
    max_dist_v2 = 0
    
    while q2:
        u = q2.popleft()
        max_dist_v2 = max(max_dist_v2, dist2[u])
        for v in adj2[u]:
            if n1 + 1 <= v <= n1 + n2 and dist2[v] == -1:
                dist2[v] = dist2[u] + 1
                q2.append(v)
                
    max_d1 = 0
    for i in range(1, n1 + 1):
        if dist1[i] != -1:
            max_d1 = max(max_d1, dist1[i])
            
    max_d2 = 0
    for i in range(n1 + 1, n1 + n2 + 1):
        if dist2[i] != -1:
            max_d2 = max(max_d2, dist2[i])
            
    max_shortest_path = 0
    for u in range(1, n1 + 1):
        if dist1[u] == -1:
            continue
        for v in range(n1 + 1, n1 + n2 + 1):
            if dist2[v] == -1:
                continue
            current_distance = dist1[u] + 1 + dist2[v]
            max_shortest_path = max(max_shortest_path, current_distance)
            
    print(max_shortest_path)

if __name__ == '__main__':
    solve()