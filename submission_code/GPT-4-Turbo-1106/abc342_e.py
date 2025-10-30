import sys
import heapq

def read_ints():
    return list(map(int, sys.stdin.readline().split()))

def main():
    N, M = read_ints()
    trains = [read_ints() for _ in range(M)]
    
    # Create a dictionary to store all the trains departing from each station
    departures = {i: [] for i in range(1, N+1)}
    for l, d, k, c, A, B in trains:
        for i in range(k):
            t = l + i * d
            heapq.heappush(departures[A], (-t, t + c, B))
    
    # Initialize the latest arrival times to -infinity
    latest_arrival = [-float('inf')] * (N+1)
    latest_arrival[N] = float('inf')  # We are already at the destination
    
    # Process stations in reverse order
    for station in range(N-1, 0, -1):
        while departures[station]:
            # Get the latest train from the current station
            neg_dep_time, arr_time, dest = heapq.heappop(departures[station])
            dep_time = -neg_dep_time
            # If the arrival time of this train is before the latest arrival time of the destination
            if arr_time <= latest_arrival[dest]:
                # Update the latest arrival time for the current station
                latest_arrival[station] = max(latest_arrival[station], dep_time)
    
    # Output the results
    for station in range(1, N):
        if latest_arrival[station] == -float('inf'):
            print("Unreachable")
        else:
            print(latest_arrival[station])

if __name__ == "__main__":
    main()