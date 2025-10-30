# YOUR CODE HERE
import heapq
import sys

def solve():
    # Read N, A, B, C from the first line
    N, A, B, C = map(int, sys.stdin.readline().split())
    
    # Read the D matrix (D[i][j] is the distance between city i+1 and city j+1)
    D = []
    for _ in range(N):
        D.append(list(map(int, sys.stdin.readline().split())))

    # Dijkstra's algorithm implementation
    # start_node: The 0-indexed city ID from which to calculate shortest paths.
    # cost_type: A string ('car' or 'train') to determine edge cost calculation.
    def dijkstra(start_node, cost_type):
        # Initialize distances to all nodes as infinity
        dist = [float('inf')] * N
        # Distance to the start node is 0
        dist[start_node] = 0
        
        # Priority queue stores tuples of (cost, city_id)
        # It is a min-heap, so the smallest cost path is always at the top
        pq = [(0, start_node)]

        while pq:
            current_cost, u = heapq.heappop(pq)

            # If we've found a shorter path to u already, skip this one
            # This check is crucial for performance and correctness
            if current_cost > dist[u]:
                continue

            # Explore neighbors of u
            for v in range(N):
                # Skip self-loops. D_ii is 0, implying no travel cost to self.
                # The problem constraints specify D_ij > 0 for i != j.
                if u == v:
                    continue 

                d_uv = D[u][v] # Distance value from D matrix
                
                # Calculate edge cost based on transportation type
                if cost_type == 'car':
                    edge_cost = d_uv * A
                elif cost_type == 'train':
                    edge_cost = d_uv * B + C
                else:
                    # This case should not be reached with valid cost_type inputs
                    raise ValueError("Invalid cost_type specified for Dijkstra.")

                # If a shorter path to v is found through u
                if dist[u] + edge_cost < dist[v]:
                    dist[v] = dist[u] + edge_cost
                    heapq.heappush(pq, (dist[v], v))
        return dist

    # Step 1: Calculate shortest paths from City 1 (0-indexed: 0) to all other cities using only the company car.
    # dist_car[k] will store the minimum time to reach city k (0-indexed) from city 0 by car.
    dist_car = dijkstra(0, 'car')

    # Step 2: Calculate shortest paths from all cities to City N (0-indexed: N-1) using only the train.
    # This is achieved by running Dijkstra from City N (N-1) and finding shortest paths to all other cities.
    # Since the D matrix is symmetric (D_ij = D_ji), the graph is undirected, so distances from k to N-1
    # are the same as N-1 to k.
    # dist_train[k] will store the minimum time to reach city N-1 from city k (0-indexed) by train.
    dist_train = dijkstra(N - 1, 'train')

    # Step 3: Find the minimum total time.
    # Iterate through all possible cities 'k' (0-indexed) where a switch from car to train can occur.
    # The total time for a path switching at city 'k' is (time from City 1 to City k by car) + (time from City k to City N by train).
    min_total_time = float('inf')
    for k in range(N):
        # dist_car[k] is the min time to get from City 1 to City (k+1) by car.
        # dist_train[k] is the min time to get from City (k+1) to City N by train.
        total_time_at_k = dist_car[k] + dist_train[k]
        min_total_time = min(min_total_time, total_time_at_k)
        
    # Print the final minimum time
    print(min_total_time)

# Call the solve function to run the program
solve()