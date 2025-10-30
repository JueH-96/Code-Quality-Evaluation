import collections

def solve():
    n, m = map(int, input().split())
    bridges_info = []
    for i in range(m):
        u, v, t = map(int, input().split())
        bridges_info.append(((u, v), t))
    
    q = int(input())
    queries = []
    for _ in range(q):
        k = int(input())
        required_bridge_indices = list(map(int, input().split()))
        queries.append(required_bridge_indices)
        
    results = []
    for required_bridge_indices in queries:
        required_bridges = [bridges_info[i-1] for i in required_bridge_indices]
        min_total_time = float('inf')
        
        import itertools
        
        permutations = list(itertools.permutations(range(len(required_bridges))))
        directions_options = list(itertools.product([0, 1], repeat=len(required_bridges)))
        
        for p_index in permutations:
            for directions in directions_options:
                current_path_time = 0
                current_island = 0 # start at island 1 (index 0)
                last_island = 0
                
                path_segments_time = 0
                
                sequence_of_bridges = []
                sequence_of_islands = [1]
                
                for i in range(len(required_bridges)):
                    bridge_index_in_required = p_index[i]
                    bridge_data = required_bridges[bridge_index_in_required]
                    (u, v), time = bridge_data
                    direction = directions[i]
                    
                    start_node_bridge = 0
                    end_node_bridge = 0
                    if direction == 0:
                        start_node_bridge = u
                        end_node_bridge = v
                    else:
                        start_node_bridge = v
                        end_node_bridge = u
                        
                    source_island = last_island
                    destination_island = start_node_bridge - 1
                    
                    distances = collections.defaultdict(lambda: float('inf'))
                    for i_node in range(1, n + 1):
                        distances[i_node] = float('inf')
                    distances[source_island+1] = 0
                    
                    import heapq
                    pq = [(0, source_island+1)]
                    
                    while pq:
                        d, current_node = heapq.heappop(pq)
                        if d > distances[current_node]:
                            continue
                        for bridge_idx in range(m):
                            (u_bridge, v_bridge), t_bridge = bridges_info[bridge_idx]
                            if u_bridge == current_node:
                                neighbor = v_bridge
                                weight = t_bridge
                                if distances[current_node] + weight < distances[neighbor]:
                                    distances[neighbor] = distances[current_node] + weight
                                    heapq.heappush(pq, (distances[neighbor], neighbor))
                            elif v_bridge == current_node:
                                neighbor = u_bridge
                                weight = t_bridge
                                if distances[current_node] + weight < distances[neighbor]:
                                    distances[neighbor] = distances[current_node] + weight
                                    heapq.heappush(pq, (distances[neighbor], neighbor))
                                    
                    segment_path_cost = distances[destination_island+1]
                    if segment_path_cost == float('inf'):
                        segment_path_cost = 0
                        
                    path_segments_time += segment_path_cost
                    path_segments_time += time
                    last_island = end_node_bridge - 1
                    
                distances_to_end = collections.defaultdict(lambda: float('inf'))
                for i_node in range(1, n + 1):
                    distances_to_end[i_node] = float('inf')
                distances_to_end[last_island+1] = 0
                pq_end = [(0, last_island+1)]
                while pq_end:
                    d, current_node = heapq.heappop(pq_end)
                    if d > distances_to_end[current_node]:
                        continue
                    for bridge_idx in range(m):
                        (u_bridge, v_bridge), t_bridge = bridges_info[bridge_idx]
                        if u_bridge == current_node:
                            neighbor = v_bridge
                            weight = t_bridge
                            if distances_to_end[current_node] + weight < distances_to_end[neighbor]:
                                distances_to_end[neighbor] = distances_to_end[current_node] + weight
                                heapq.heappush(pq_end, (distances_to_end[neighbor], neighbor))
                        elif v_bridge == current_node:
                            neighbor = u_bridge
                            weight = t_bridge
                            if distances_to_end[current_node] + weight < distances_to_end[neighbor]:
                                distances_to_end[neighbor] = distances_to_end[current_node] + weight
                                heapq.heappush(pq_end, (distances_to_end[neighbor], neighbor))
                                
                segment_path_to_n_cost = distances_to_end[n]
                if segment_path_to_n_cost == float('inf'):
                    segment_path_to_n_cost = 0

                total_cost = path_segments_time + segment_path_to_n_cost
                min_total_time = min(min_total_time, total_cost)
                
        results.append(min_total_time)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()