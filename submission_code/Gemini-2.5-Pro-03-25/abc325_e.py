# YOUR CODE HERE
import heapq
import sys

# Read input using sys.stdin.readline for potentially faster I/O
input = sys.stdin.readline

def solve():
    # Read problem parameters: N cities, A, B, C coefficients for travel times
    N, A, B, C = map(int, input().split())
    
    # Read the distance matrix D. D[i][j] is the distance between city i+1 and city j+1.
    # Store it as a 0-indexed list of lists.
    D = []
    for _ in range(N):
        D.append(list(map(int, input().split())))

    # State representation for Dijkstra:
    # We use a graph with 2N nodes.
    # Node index i (where 0 <= i < N) represents being at city i (0-based index) and having arrived/currently using the company car mode.
    # Node index N+i (where 0 <= i < N) represents being at city i (0-based index) and having arrived/currently using the train mode.
    # City k (1-based index) corresponds to 0-based index k-1.

    # Initialize distance array `dist` for all 2N states to infinity.
    dist = [float('inf')] * (2 * N)
    
    # The starting state is city 1 (index 0) in car mode. This corresponds to node index 0.
    # The distance to the starting state is 0.
    start_node_idx = 0 
    dist[start_node_idx] = 0
    
    # Priority queue `pq` for Dijkstra's algorithm. Stores tuples (current_distance, node_index).
    # Min-heap property based on distance.
    pq = [(0, start_node_idx)] 

    # Main loop of Dijkstra's algorithm
    while pq:
        # Extract the node `u` with the smallest distance `d` from the priority queue.
        d, u = heapq.heappop(pq)

        # Optimization: If the extracted distance `d` is greater than the already known shortest distance to `u`,
        # this means we have found a shorter path to `u` earlier. Skip processing this element.
        if d > dist[u]:
            continue

        # Process node `u`. Check if it represents a 'car mode' state or 'train mode' state.
        if u < N:
            # Current state is 'car mode'. The city index (0-based) is `u`.
            curr_city_idx = u
            
            # Explore possible next moves from this state:

            # Option 1: Travel by company car to another city `next_city_idx`.
            for next_city_idx in range(N):
                # Traveling to the same city takes 0 time (D[i][i] = 0), but it's generally not useful. Skip it explicitly.
                if curr_city_idx == next_city_idx:
                    continue
                
                # Calculate the cost of travel from `curr_city_idx` to `next_city_idx` by car.
                cost = D[curr_city_idx][next_city_idx] * A
                # The destination state is also in 'car mode'. The node index is `next_city_idx`.
                neighbor_node_idx = next_city_idx
                
                # Relaxation step: If path through `u` to `neighbor_node_idx` is shorter, update distance and push to queue.
                if dist[u] + cost < dist[neighbor_node_idx]:
                    dist[neighbor_node_idx] = dist[u] + cost
                    heapq.heappush(pq, (dist[neighbor_node_idx], neighbor_node_idx))

            # Option 2: Switch to train mode at the current city `curr_city_idx`.
            # This switch takes 0 time.
            cost_switch = 0 
            # The destination state is the same city `curr_city_idx`, but now in 'train mode'.
            # The node index for this state is `curr_city_idx + N`.
            neighbor_node_idx_switch = curr_city_idx + N 
            
            # Relaxation step for the switch transition: Update distance if this path to the 'train mode' state is shorter.
            if dist[u] + cost_switch < dist[neighbor_node_idx_switch]:
                 dist[neighbor_node_idx_switch] = dist[u] + cost_switch
                 heapq.heappush(pq, (dist[neighbor_node_idx_switch], neighbor_node_idx_switch))

        else:
            # Current state is 'train mode'. The city index (0-based) is `u - N`.
            curr_city_idx = u - N
            
            # Explore possible next moves from this state:
            # Only option is to travel by train, as switching back to car is not allowed.

            # Option: Travel by train to another city `next_city_idx`.
            for next_city_idx in range(N):
                 # Skip traveling to the same city.
                 if curr_city_idx == next_city_idx:
                     continue

                 # Calculate the cost of travel from `curr_city_idx` to `next_city_idx` by train.
                 cost = D[curr_city_idx][next_city_idx] * B + C
                 # The destination state is also in 'train mode'.
                 # The node index for this state is `next_city_idx + N`.
                 neighbor_node_idx = next_city_idx + N
                 
                 # Relaxation step: If path through `u` to `neighbor_node_idx` is shorter, update distance and push to queue.
                 if dist[u] + cost < dist[neighbor_node_idx]:
                     dist[neighbor_node_idx] = dist[u] + cost
                     heapq.heappush(pq, (dist[neighbor_node_idx], neighbor_node_idx))

    # After Dijkstra finishes, `dist` array contains the shortest times to reach each state (city, mode).
    # The goal is to reach city N (which has 0-based index N-1).
    # We need the minimum time to reach city N, regardless of whether the final leg was by car or train.
    # The minimum time is the minimum of:
    # - distance to state (city N, car mode), which is node index `N-1`.
    # - distance to state (city N, train mode), which is node index `N-1 + N = 2N-1`.
    result = min(dist[N-1], dist[N-1 + N])
    
    # The problem asks for the answer as an integer. Since all costs are derived from integer inputs,
    # the minimum distance will be an integer (or infinity if unreachable, but constraints guarantee reachability).
    # Explicitly cast to integer for safety / adherence to output format.
    print(int(result))

# Call the main function to run the solution
solve()