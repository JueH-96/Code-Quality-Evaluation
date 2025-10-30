import sys
import itertools

def solve():
    n, m = map(int, sys.stdin.readline().split())
    bridges_info = []
    for _ in range(m):
        u, v, t = map(int, sys.stdin.readline().split())
        bridges_info.append(((u, v), t))
    q = int(sys.stdin.readline())
    queries = []
    for _ in range(q):
        k = int(sys.stdin.readline())
        required_bridge_indices = list(map(int, sys.stdin.readline().split()))
        queries.append(required_bridge_indices)
    
    shortest_paths = [[float('inf')] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        shortest_paths[i][i] = 0
    for i in range(m):
        (u, v), t = bridges_info[i]
        shortest_paths[u][v] = min(shortest_paths[u][v], t)
        shortest_paths[v][u] = min(shortest_paths[v][u], t)
        
    for k_island in range(1, n + 1):
        for i_island in range(1, n + 1):
            for j_island in range(1, n + 1):
                if shortest_paths[i_island][k_island] != float('inf') and shortest_paths[k_island][j_island] != float('inf'):
                    shortest_paths[i_island][j_island] = min(shortest_paths[i_island][j_island], shortest_paths[i_island][k_island] + shortest_paths[k_island][j_island])
                    
    results = []
    for required_bridge_indices in queries:
        required_bridges = []
        for index in required_bridge_indices:
            required_bridges.append(bridges_info[index-1])
        
        min_total_time = float('inf')
        permutations = list(itertools.permutations(required_bridges))
        
        for bridge_permutation in permutations:
            num_required_bridges = len(bridge_permutation)
            for directions_config in itertools.product([0, 1], repeat=num_required_bridges):
                current_time = 0
                current_island = 1
                
                path_segments = []
                last_end_island = 1
                
                for i in range(num_required_bridges):
                    bridge_info = bridge_permutation[i]
                    (u, v), time = bridge_info
                    direction = directions_config[i]
                    start_island_bridge = u if direction == 0 else v
                    end_island_bridge = v if direction == 0 else u
                    
                    path_time = shortest_paths[last_end_island][start_island_bridge]
                    if path_time == float('inf'):
                        current_time = float('inf')
                        break
                    current_time += path_time
                    current_time += time
                    last_end_island = end_island_bridge
                    
                if current_time != float('inf'):
                    final_path_time = shortest_paths[last_end_island][n]
                    if final_path_time != float('inf'):
                        current_time += final_path_time
                        min_total_time = min(min_total_time, current_time)
                        
        results.append(min_total_time if min_total_time != float('inf') else -1)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()