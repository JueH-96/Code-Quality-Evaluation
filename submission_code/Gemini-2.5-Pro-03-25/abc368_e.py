# YOUR CODE HERE
import heapq
import sys
import bisect

# Increase recursion depth limit if necessary, although this solution is iterative.
# sys.setrecursionlimit(200000) 

def solve():
    # Read input: N cities, M trains, fixed delay X1 for train 1
    N, M, X1 = map(int, sys.stdin.readline().split())
    
    # List to store train information. Each element is [A_i, B_i, S_i, T_i]
    # We use 0-based indexing for trains internally, so train 1 is index 0, train M is index M-1.
    trains = []
    for _ in range(M):
        trains.append(list(map(int, sys.stdin.readline().split()))) 

    # Preprocess train data for efficient neighbor lookup.
    # Store departure information for each city.
    # departures_by_city[C] = list of tuples (departure_time, train_idx) 
    # where train_idx is the 0-based index of the train in the `trains` list.
    # We use 1-based indexing for cities (1 to N) as given in the problem.
    departures_by_city = [[] for _ in range(N + 1)]
    for idx in range(M):
        A_i, B_i, S_i, T_i = trains[idx]
        # Add departure event (time, train index) to the list for the departure city A_i.
        departures_by_city[A_i].append((S_i, idx)) 

    # Sort the departure lists for each city based on departure time.
    # This allows efficient searching using binary search (bisect_left).
    for C in range(1, N + 1):
        departures_by_city[C].sort()

    # The problem asks to find non-negative integers X_2, ..., X_M to minimize their sum,
    # subject to constraints: T_i + X_i <= S_j + X_j for all transferable pairs (i, j).
    # A pair (i, j) is transferable if B_i = A_j and T_i <= S_j.
    # Let's reformulate the constraints: X_j >= X_i + T_i - S_j.
    # Let Y_k = -X_k. The constraints become -Y_j >= -Y_i + T_i - S_j, which simplifies to
    # Y_j <= Y_i + S_j - T_i. Let w'_{ij} = S_j - T_i. Then Y_j <= Y_i + w'_{ij}.
    # Since T_i <= S_j for transferable pairs, the weights w'_{ij} >= 0.
    # We also have X_k >= 0, which means Y_k <= 0.
    # We are given X_1, so Y_1 = -X_1.
    # We want to minimize sum(X_k for k=2..M) = sum(-Y_k for k=2..M), which is equivalent to maximizing sum(Y_k for k=2..M).
    # This is a shortest path problem structure on a graph where nodes are trains (indices 0 to M-1).
    # We use Dijkstra's algorithm because all edge weights w'_{ij} are non-negative.

    # `d[k]` will store the shortest path distance found so far for node k, conceptually related to Y_k.
    # Initialize distances to infinity.
    d = [float('inf')] * M
    
    # Apply the constraint Y_1 = -X1. Since X1 >= 1, -X1 is negative or zero.
    # Initialize the distance for the first train (index 0).
    if M > 0: # Constraints state M >= 2. This check is for safety.
        d[0] = -X1 
    
    # Priority queue for Dijkstra's algorithm. Stores tuples (distance, node_idx).
    # heapq implements a min-heap, which is what Dijkstra needs for shortest paths.
    pq = []
    if M > 0:
       heapq.heappush(pq, (d[0], 0)) # Start Dijkstra from train 1 (index 0)

    # Main loop of Dijkstra's algorithm
    while pq:
        # Extract node with the smallest distance from the priority queue.
        dist, curr_node_idx = heapq.heappop(pq)

        # If the extracted distance is greater than the already known shortest distance,
        # it means we found a shorter path earlier. Skip this element.
        if dist > d[curr_node_idx]:
            continue

        # Get information about the current train (node `curr_node_idx`).
        A_i, B_i, S_i, T_i = trains[curr_node_idx]
        
        arrival_city = B_i
        arrival_time = T_i # This corresponds to T_i in the weight calculation w'_{ij} = S_j - T_i

        # Check city index validity (should be between 1 and N)
        if not (1 <= arrival_city <= N):
             continue # Precaution, should not happen based on constraints.

        # Find potential transfers: trains departing from `arrival_city`.
        departures = departures_by_city[arrival_city]
        
        # If no trains depart from this city, there are no outgoing edges from this node.
        if not departures: 
             continue

        # Use `bisect_left` for efficient search. Find the index of the first train `j`
        # departing from `arrival_city` at time S_j >= `arrival_time` (T_i).
        # The search key is `(arrival_time, -1)`. Tuple comparison primarily uses the first element.
        # `bisect_left` finds the insertion point, which is the index of the first element >= key.
        key = arrival_time
        first_valid_idx = bisect.bisect_left(departures, (key, -1)) 

        # Iterate over all trains departing at or after arrival_time (these are the neighbors).
        for k in range(first_valid_idx, len(departures)):
            next_dep_time, next_node_idx = departures[k] # S_j and index j
            
            # Calculate the weight of the edge from `curr_node_idx` (i) to `next_node_idx` (j).
            # Weight w'_{ij} = S_j - T_i. This is guaranteed non-negative.
            weight = next_dep_time - arrival_time 
            
            # Calculate the potential new shortest distance to the neighbor node.
            new_dist = d[curr_node_idx] + weight
            
            # If this path through `curr_node_idx` is shorter than the currently known shortest path
            # to `next_node_idx`, update the distance and add/update the neighbor in the priority queue.
            if new_dist < d[next_node_idx]:
                d[next_node_idx] = new_dist
                heapq.heappush(pq, (new_dist, next_node_idx))

    # After Dijkstra finishes, `d[k]` holds the computed shortest path distance for node k.
    # Now, convert these distances back to the required X_k values.
    # Recall Y_k = -X_k. We need Y_k <= 0 (from X_k >= 0).
    # The shortest path computation gives candidate values for Y_k. We must enforce Y_k <= 0.
    # So, the actual value is Y_k = min(0, d[k]).
    # Then X_k = -Y_k = -min(0, d[k]). This simplifies to X_k = max(0, -d[k]).
    
    X = [0] * M
    for k in range(M):
        if d[k] == float('inf'):
             # If node k is unreachable from node 0 (train 1), its delay X_k is not directly
             # influenced by X1. The minimum non-negative value satisfying local constraints is X_k = 0.
             # This minimizes the sum.
             X[k] = 0
        else:
             # Compute X_k using the formula derived from the transformation and constraints.
             # X_k = max(0, -d[k]) ensures X_k is non-negative.
             # Ensure the result is an integer. Since all inputs are integers, calculations
             # involving additions and subtractions maintain integrality. -d[k] is integer.
             X[k] = int(max(0, -d[k])) 
             
    # The problem asks for X_2, ..., X_M. These correspond to indices 1 to M-1 in our 0-based list X.
    # Check if M > 1 because if M=1, there is no X_2 to print. Constraints say M >= 2.
    if M > 1:
      # Print the values X[1], X[2], ..., X[M-1] separated by spaces.
      print(*(X[1:]))
    # No explicit else needed as M >= 2 guarantees M > 1.

# Execute the main solve function
solve()