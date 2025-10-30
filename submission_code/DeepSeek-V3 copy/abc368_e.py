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
    
    # Create adjacency list for transfers
    adj = [[] for _ in range(M)]
    for i in range(M):
        for j in range(M):
            if trains[i][1] == trains[j][0] and trains[i][3] <= trains[j][2]:
                adj[i].append(j)
    
    # Use a priority queue to process the trains in order of their arrival times
    # We need to ensure that for each transfer, the condition is satisfied
    # We can process the trains in topological order based on the transfer dependencies
    # However, since the graph is not necessarily a DAG, we need to handle it differently
    
    # Instead, we can use a greedy approach where we process the trains in order of their arrival times
    # and for each train, we set its X_i to be the maximum required to satisfy all its incoming transfers
    
    # To do this, we need to process the trains in a way that ensures all dependencies are processed before the current train
    # One way is to process the trains in order of their arrival times, but this may not be sufficient
    
    # Alternatively, we can use a BFS-like approach where we process the trains in levels based on their dependencies
    
    # Here, we will use a priority queue to process the trains in order of their arrival times
    # and for each train, we will update its X_i based on the X_j of all trains that can transfer to it
    
    # Initialize a priority queue with the first train
    heap = []
    heapq.heappush(heap, (trains[0][3], 0))
    
    # Process the heap
    while heap:
        current_time, u = heapq.heappop(heap)
        for v in adj[u]:
            # Calculate the required X_v
            required_X_v = max(0, (trains[u][3] + X[u]) - trains[v][2])
            if X[v] < required_X_v:
                X[v] = required_X_v
                heapq.heappush(heap, (trains[v][3], v))
    
    # Print X2 to XM
    print(' '.join(map(str, X[1:])))

if __name__ == "__main__":
    main()