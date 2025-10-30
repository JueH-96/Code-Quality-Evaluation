import heapq

def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, t = map(int, input().split())
        edges.append((u, v, t))
    
    q = int(input())
    for _ in range(q):
        k = int(input())
        required_bridges = list(map(int, input().split()))
        
        def calculate_min_time(start_node, end_node, required_bridges_indices):
            
            import heapq
            
            graph = {i: [] for i in range(1, n + 1)}
            for i in range(m):
                u, v, t = edges[i]
                graph[u].append((v, t, i+1))
                graph[v].append((u, t, i+1))
            
            
            def find_shortest_path_with_required_edges(start, end, required_edges):
                
                
                pq = [(0, start, set())]  # (cost, node, used_edges)
                visited = set()
                
                while pq:
                    cost, current_node, used_edges = heapq.heappop(pq)
                    
                    if (current_node, tuple(sorted(list(used_edges)))) in visited:
                        continue
                    visited.add((current_node, tuple(sorted(list(used_edges)))))
                    
                    if current_node == end and set(required_edges) == used_edges:
                        return cost
                    
                    for neighbor, weight, edge_index in graph[current_node]:
                        new_used_edges = set(used_edges)
                        if edge_index in required_edges:
                            new_used_edges.add(edge_index)
                        
                        heapq.heappush(pq, (cost + weight, neighbor, new_used_edges))
                
                return float('inf')
            
            
            min_cost = find_shortest_path_with_required_edges(start_node, end_node, required_bridges_indices)
            return min_cost

        ans = calculate_min_time(1, n, required_bridges)
        print(ans)

solve()