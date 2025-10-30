import collections

def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        u, v, w = map(int, input().split())
        edges.append(((u, v), w))
    
    adj = collections.defaultdict(list)
    total_edge_length = 0
    for (u, v), weight in edges:
        adj[u].append((v, weight))
        adj[v].append((u, weight))
        total_edge_length += weight
        
    def bfs(start_node):
        distances = {}
        for i in range(1, n + 1):
            distances[i] = float('inf')
        distances[start_node] = 0
        queue = collections.deque([start_node])
        while queue:
            u = queue.popleft()
            for v, weight in adj[u]:
                if distances[v] == float('inf'):
                    distances[v] = distances[u] + weight
                    queue.append(v)
        return distances
        
    distances_from_1 = bfs(1)
    farthest_node_from_1 = -1
    max_distance = -1
    for i in range(1, n + 1):
        if distances_from_1[i] > max_distance:
            max_distance = distances_from_1[i]
            farthest_node_from_1 = i
            
    distances_from_farthest = bfs(farthest_node_from_1)
    diameter = 0
    for i in range(1, n + 1):
        diameter = max(diameter, distances_from_farthest[i])
        
    result = 2 * total_edge_length - diameter
    print(result)

if __name__ == '__main__':
    solve()