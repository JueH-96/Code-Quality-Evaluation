import collections

def solve():
    n1, n2, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    n = n1 + n2
    adj = collections.defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    def get_distances(start_node):
        distances = collections.defaultdict(lambda: float('inf'))
        distances[start_node] = 0
        queue = collections.deque([start_node])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if distances[v] == float('inf'):
                    distances[v] = distances[u] + 1
                    queue.append(v)
        return distances
        
    distances_from_1 = get_distances(1)
    distances_from_n_plus_n2 = get_distances(n)
    
    max_dist_1 = 0
    for i in range(1, n1 + 1):
        if distances_from_1[i] != float('inf'):
            max_dist_1 = max(max_dist_1, distances_from_1[i])
            
    max_dist_2 = 0
    for i in range(n1 + 1, n + 1):
        if distances_from_n_plus_n2[i] != float('inf'):
            max_dist_2 = max(max_dist_2, distances_from_n_plus_n2[i])
            
    max_shortest_path = 1 + max_dist_1 + max_dist_2
    print(max_shortest_path)

if __name__ == '__main__':
    solve()