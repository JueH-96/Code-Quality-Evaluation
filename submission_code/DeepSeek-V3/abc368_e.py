import sys
import heapq

def main():
    N, M, X1 = map(int, sys.stdin.readline().split())
    trains = []
    for _ in range(M):
        A, B, S, T = map(int, sys.stdin.readline().split())
        trains.append((A, B, S, T))
    
    # Initialize X array
    X = [0] * M
    X[0] = X1
    
    # Create a graph where edges represent the transfer constraints
    # We need to find the minimal X_i for each train such that the constraints are satisfied
    # We can model this as a shortest path problem where X_i is the distance from the source
    
    # We will use a priority queue to process the trains in order of their arrival times
    # Initialize the queue with the first train
    heap = []
    heapq.heappush(heap, (trains[0][3] + X[0], 0))
    
    # To keep track of the latest possible departure time for each city
    latest_departure = {}
    
    while heap:
        current_time, train_idx = heapq.heappop(heap)
        A, B, S, T = trains[train_idx]
        arrival_time = T + X[train_idx]
        
        # Update the latest departure time for city B
        if B in latest_departure:
            if arrival_time > latest_departure[B]:
                latest_departure[B] = arrival_time
        else:
            latest_departure[B] = arrival_time
        
        # Now, for all trains that depart from B and have S_j >= T_i
        # We need to ensure that S_j + X_j >= arrival_time
        # So, X_j >= arrival_time - S_j
        # We can iterate over all trains that depart from B and have S_j >= T_i
        # Since the trains are not sorted, we need to find them efficiently
        # To do this, we can preprocess the trains and group them by departure city
        # So, we need to create a dictionary where the key is the departure city and the value is a list of tuples (S_j, j)
        
        # Preprocessing step: create a dictionary to map departure city to list of (S_j, j)
        if not hasattr(main, 'departure_map'):
            main.departure_map = {}
            for idx, (A_j, B_j, S_j, T_j) in enumerate(trains):
                if A_j not in main.departure_map:
                    main.departure_map[A_j] = []
                main.departure_map[A_j].append((S_j, idx))
        
        if B in main.departure_map:
            for S_j, j in main.departure_map[B]:
                if S_j >= T:
                    # Calculate the required X_j
                    required_X_j = max(0, arrival_time - S_j)
                    if X[j] < required_X_j:
                        X[j] = required_X_j
                        heapq.heappush(heap, (trains[j][3] + X[j], j))
    
    # Now, we need to ensure that all constraints are satisfied
    # We can iterate over all pairs of trains and check the condition
    # However, this is O(M^2) which is too slow for M=2e5
    # Instead, we can trust that the above process has correctly set the X_i values
    # Since the problem guarantees a unique solution, we can proceed
    
    # Output X_2 to X_M
    print(' '.join(map(str, X[1:])))

if __name__ == "__main__":
    main()