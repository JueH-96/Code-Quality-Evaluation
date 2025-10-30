def solve():
    n, m = map(int, input().split())
    x = list(map(int, input().split()))
    
    def get_cycle_distance(u, v, n_islands):
        return min(abs(u - v), n_islands - abs(u - v))
        
    original_tour_length = 0
    for i in range(m - 1):
        original_tour_length += get_cycle_distance(x[i], x[i+1], n)
        
    min_tour_length_after_removal = original_tour_length
    
    for bridge_to_remove in range(1, n + 1):
        current_tour_length = 0
        for i in range(m - 1):
            u = x[i]
            v = x[i+1]
            
            def get_shortest_path_length(start_node, end_node, removed_bridge_index):
                if start_node == end_node:
                    return 0
                distance = [-1] * (n + 1)
                distance[start_node] = 0
                queue = [start_node]
                
                while queue:
                    current_node = queue.pop(0)
                    if current_node == end_node:
                        return distance[end_node]
                        
                    neighbors = []
                    if current_node > 1:
                        neighbors.append(current_node - 1)
                    if current_node < n:
                        neighbors.append(current_node + 1)
                    neighbors.append(n if current_node == 1 else 1 if current_node == n else -1)
                    
                    valid_neighbors = []
                    for neighbor in neighbors:
                        if neighbor != -1:
                            bridge_connection = tuple(sorted((current_node, neighbor)))
                            bridge_index = -1
                            if bridge_connection == tuple(sorted((1, 2))):
                                bridge_index = 1
                            elif bridge_connection == tuple(sorted((2, 3))):
                                bridge_index = 2
                            elif bridge_connection == tuple(sorted((3, 4))):
                                bridge_index = 3
                            elif bridge_connection == tuple(sorted((n-1, n))):
                                bridge_index = n-1
                            elif bridge_connection == tuple(sorted((1, n))):
                                bridge_index = n
                            else:
                                bridge_index = min(bridge_connection) if bridge_connection[1] == bridge_connection[0] + 1 else -2
                                
                            is_bridge_to_be_removed = False
                            if 1 <= removed_bridge_index <= n - 1:
                                if bridge_connection == tuple(sorted((removed_bridge_index, removed_bridge_index + 1))):
                                    is_bridge_to_be_removed = True
                            elif removed_bridge_index == n:
                                if bridge_connection == tuple(sorted((1, n))):
                                    is_bridge_to_be_removed = True

                            if not is_bridge_to_be_removed:
                                valid_neighbors.append(neighbor)
                                
                    for neighbor in valid_neighbors:
                        if distance[neighbor] == -1:
                            distance[neighbor] = distance[current_node] + 1
                            queue.append(neighbor)
                            
                return float('inf')

            shortest_path = get_shortest_path_length(u, v, bridge_to_remove)
            if shortest_path == float('inf'):
                # Should not happen in this problem context since graph remains connected (for N>=3)
                pass 
            else:
                current_tour_length += shortest_path
                
        min_tour_length_after_removal = min(min_tour_length_after_removal, current_tour_length)
        
    print(min_tour_length_after_removal)

if __name__ == '__main__':
    solve()