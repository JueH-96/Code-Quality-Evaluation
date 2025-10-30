from collections import defaultdict
import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    N, M = map(int, data[0].split())
    
    # Adjacency list for trains
    trains = defaultdict(list)
    
    for i in range(1, M + 1):
        l, d, k, c, A, B = map(int, data[i].split())
        for j in range(k):
            t = l + j * d
            trains[A].append((t, c, B))
    
    # Result array
    results = [-float('inf')] * (N + 1)
    results[1] = float('inf')  # f(1) is infinity since we can start at any time
    
    # Priority queue for processing the stations
    pq = []
    heapq.heappush(pq, (-results[1], 1))  # (negative time, station)
    
    while pq:
        current_time, station = heapq.heappop(pq)
        current_time = -current_time
        
        for dep_time, travel_time, next_station in trains[station]:
            if dep_time <= current_time:
                arrival_time = dep_time + travel_time
                if arrival_time > results[next_station]:
                    results[next_station] = arrival_time
                    heapq.heappush(pq, (-arrival_time, next_station))
    
    # Output results
    output = []
    for i in range(2, N + 1):
        if results[i] == -float('inf'):
            output.append("Unreachable")
        else:
            output.append(str(results[i]))
    
    print("
".join(output))

if __name__ == "__main__":
    main()