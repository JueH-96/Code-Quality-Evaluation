import collections

def solve():
    n, m, k = map(int, input().split())
    edges_input = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges_input.append(((u, v), w))
    
    min_cost = float('inf')
    edge_indices = list(range(m))
    
    import itertools
    
    combinations_indices = itertools.combinations(edge_indices, n - 1)
    
    for indices in combinations_indices:
        selected_edges_data = [edges_input[i] for i in indices]
        selected_edges = [edge_data[0] for edge_data in selected_edges_data]
        edge_weights = [edge_data[1] for edge_data in selected_edges_data]
        
        adj = collections.defaultdict(list)
        for i in range(n - 1):
            u, v = selected_edges[i]
            adj[u].append(v)
            adj[v].append(u)
            
        visited = set()
        
        def dfs(u):
            visited.add(u)
            for v in adj[u]:
                if v not in visited:
                    dfs(v)
                    
        dfs(1)
        
        is_spanning_tree = False
        if len(visited) == n:
            is_spanning_tree = True
            
        if is_spanning_tree:
            current_cost = sum(edge_weights) % k
            min_cost = min(min_cost, current_cost)
            
    print(min_cost)

if __name__ == '__main__':
    solve()