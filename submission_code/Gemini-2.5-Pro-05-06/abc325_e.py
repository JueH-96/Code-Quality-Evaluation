import sys

def main():
    N, A, B, C = map(int, sys.stdin.readline().split())
    
    D_matrix = []
    for _ in range(N):
        D_matrix.append(list(map(int, sys.stdin.readline().split())))

    # Dijkstra's algorithm O(V^2) version
    # N_nodes: number of nodes
    # adj_matrix: stores D_{i,j} values.
    # start_node_idx: 0-indexed source node
    # cost_calculator_func: a function (d_ij) -> cost for edge with distance d_ij
    def dijkstra_v2(N_nodes, adj_matrix, start_node_idx, cost_calculator_func):
        dist = [float('inf')] * N_nodes
        dist[start_node_idx] = 0
        visited = [False] * N_nodes
        
        for _ in range(N_nodes): # Main loop of Dijkstra, N_nodes iterations
            min_d_val = float('inf')
            u_node = -1
            # Select unvisited node with smallest distance
            for i_node in range(N_nodes):
                if not visited[i_node] and dist[i_node] < min_d_val:
                    min_d_val = dist[i_node]
                    u_node = i_node
            
            if u_node == -1: # All remaining unvisited_nodes are unreachable or all reachable processed
                break
                
            visited[u_node] = True
            
            # For u_node, relax its neighbors v_node
            # dist[u_node] is now finalized (it's min_d_val)
            for v_node in range(N_nodes):
                if u_node == v_node: # No self-loops for travel
                    continue
                
                edge_dist_val = adj_matrix[u_node][v_node]
                # D_i,j > 0 for i != j, and A,B,C >= 1. So travel_cost_uv > 0 for distinct i,j.
                travel_cost_uv = cost_calculator_func(edge_dist_val)
                
                # Standard relaxation condition for Dijkstra.
                # Check 'not visited[v_node]' to only update distances for unvisited nodes.
                if not visited[v_node] and dist[u_node] + travel_cost_uv < dist[v_node]:
                    dist[v_node] = dist[u_node] + travel_cost_uv
                    
        return dist

    # Calculate car costs from city 1 (index 0) to all cities
    # Lambda function to calculate cost for a car trip of distance d
    car_cost_func = lambda d: d * A
    dist_car = dijkstra_v2(N, D_matrix, 0, car_cost_func)
    
    # Calculate train costs from all cities to city N (index N-1)
    # This is equivalent to costs from city N to all cities, because D_ij = D_ji.
    # Lambda function to calculate cost for a train trip of distance d
    train_cost_func = lambda d: d * B + C
    # The Dijkstra calculates shortest paths *from* N-1 to all other nodes.
    # dist_train_from_N[k] will be shortest time from N-1 to k by train.
    # Because D_ij=D_ji, costs are symmetric, so this is also shortest time from k to N-1.
    dist_train_from_N = dijkstra_v2(N, D_matrix, N - 1, train_cost_func)
    
    min_total_time = float('inf')
    for k_node_idx in range(N): # k_node_idx is the 0-indexed city where we switch
        # dist_car[k_node_idx] is min time from city 0 to city k_node_idx by car.
        # dist_train_from_N[k_node_idx] is min time from city N-1 to city k_node_idx by train.
        # This is effectively min time from k_node_idx to N-1 by train due to symmetric distances.
        
        # Problem constraints (D_ij > 0 for i!=j) ensure graph is connected.
        # So all dists will be finite values (not float('inf')).
        cost_at_k = dist_car[k_node_idx] + dist_train_from_N[k_node_idx]
        if cost_at_k < min_total_time:
            min_total_time = cost_at_k
            
    # The problem guarantees all inputs are integers, costs A,B,C are integers.
    # D_ij are integers. So all path costs will be integers.
    # float('inf') is only for initialization. min_total_time will be a finite number.
    # Python's float representation of whole numbers (e.g., 78.0) needs int() for clean output.
    sys.stdout.write(str(int(min_total_time)) + "
")

if __name__ == '__main__':
    main()