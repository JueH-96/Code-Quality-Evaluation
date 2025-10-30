import sys
import heapq

def min_time_to_travel(N, A, B, C, D):
    # Initialize the distance matrix with infinity
    dist = [[float('inf')] * N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0
    
    # Fill the distance matrix with given distances
    for i in range(N):
        for j in range(N):
            dist[i][j] = D[i][j]
    
    # Priority queue to store the (time, current_city) tuples
    pq = [(0, 0)]
    
    while pq:
        current_time, current_city = heapq.heappop(pq)
        
        # If we have reached the destination city, return the time
        if current_city == N - 1:
            return current_time
        
        # Explore the next cities using company car
        for next_city in range(N):
            if next_city != current_city:
                next_time = current_time + dist[current_city][next_city] * A
                if next_time < dist[next_city][next_city]:
                    dist[next_city][next_city] = next_time
                    heapq.heappush(pq, (next_time, next_city))
        
        # Explore the next cities using train
        for next_city in range(N):
            if next_city != current_city:
                next_time = current_time + dist[current_city][next_city] * B + C
                if next_time < dist[next_city][next_city]:
                    dist[next_city][next_city] = next_time
                    heapq.heappush(pq, (next_time, next_city))
    
    return -1

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = int(data[1])
B = int(data[2])
C = int(data[3])

D = []
for i in range(N):
    row = list(map(int, data[4 + i * N: 4 + (i + 1) * N]))
    D.append(row)

# Calculate and print the minimum time to travel from city 1 to city N
print(min_time_to_travel(N, A, B, C, D))