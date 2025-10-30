# YOUR CODE HERE
import heapq
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    trains = []
    index = 2
    for _ in range(M):
        l = int(data[index])
        d = int(data[index + 1])
        k = int(data[index + 2])
        c = int(data[index + 3])
        A = int(data[index + 4])
        B = int(data[index + 5])
        for i in range(k):
            t = l + i * d
            trains.append((t, c, A, B))
        index += 6
    
    # Sort trains by departure time
    trains.sort()
    
    # Initialize the latest arrival times
    latest_arrival = [-float('inf')] * (N + 1)
    latest_arrival[N] = float('inf')
    
    # Priority queue for processing trains
    pq = []
    
    # Process each train in reverse order
    for t, c, A, B in reversed(trains):
        if latest_arrival[B] != -float('inf'):
            heapq.heappush(pq, (-t, A, t + c))
    
    # Process the priority queue
    while pq:
        neg_t, A, arrival_time = heapq.heappop(pq)
        t = -neg_t
        if latest_arrival[A] < t:
            latest_arrival[A] = t
            for t2, c2, A2, B2 in reversed(trains):
                if B2 == A and t2 + c2 <= t:
                    heapq.heappush(pq, (-t2, A2, t2 + c2))
    
    # Output the results
    for i in range(1, N):
        if latest_arrival[i] == -float('inf'):
            print("Unreachable")
        else:
            print(latest_arrival[i])

if __name__ == "__main__":
    main()