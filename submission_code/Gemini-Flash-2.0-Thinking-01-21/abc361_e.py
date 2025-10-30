import collections

def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        u, v, w = map(int, input().split())
        edges.append(((u, v), w))
    
    adj = collections.defaultdict(list)
    total_edge_length = 0
    for (u, v), w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
        total_edge_length += w
        
    def get_farthest_node(start_node):
        distances = [-1] * (n + 1)
        distances[start_node] = 0
        queue = collections.deque([start_node])
        farthest_node = start_node
        max_distance = 0
        
        while queue:
            u = queue.popleft()
            for v, weight in adj[u]:
                if distances[v] == -1:
                    distances[v] = distances[u] + weight
                    if distances[v] > max_distance:
                        max_distance = distances[v]
                        farthest_node = v
                    queue.append(v)
                    
        return farthest_node, max_distance
        
    farthest_node1, _ = get_farthest_node(1)
    farthest_node2, diameter = get_farthest_node(farthest_node1)
    
    result = 2 * total_edge_length - diameter
    print(result)

if __name__ == '__main__':
    solve()