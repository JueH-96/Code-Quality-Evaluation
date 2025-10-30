import heapq

def min_time(n, a, b, c, distances):
    # Initialize distances dictionary for each city with each transportation mode
    dist = {}
    dist[(1, 'Car')] = 0
    dist[(1, 'Train')] = float('inf')  # We start with a car
    for i in range(2, n+1):
        dist[(i, 'Car')] = float('inf')
        dist[(i, 'Train')] = float('inf')
    
    # Priority queue for Dijkstra's algorithm
    pq = [(0, (1, 'Car'))]
    
    while pq:
        time, (city, transport) = heapq.heappop(pq)
        
        # Skip if we've already found a better path
        if time > dist[(city, transport)]:
            continue
        
        for next_city in range(1, n+1):
            if next_city == city:
                continue
            
            d = distances[city-1][next_city-1]
            
            if transport == 'Car':
                # Continue with car
                new_time = time + d * a
                if new_time < dist[(next_city, 'Car')]:
                    dist[(next_city, 'Car')] = new_time
                    heapq.heappush(pq, (new_time, (next_city, 'Car')))
                
                # Switch to train at the current city and then go to the next city
                new_time = time + d * b + c
                if new_time < dist[(next_city, 'Train')]:
                    dist[(next_city, 'Train')] = new_time
                    heapq.heappush(pq, (new_time, (next_city, 'Train')))
            
            else:  # transport == 'Train'
                # Continue with train
                new_time = time + d * b + c
                if new_time < dist[(next_city, 'Train')]:
                    dist[(next_city, 'Train')] = new_time
                    heapq.heappush(pq, (new_time, (next_city, 'Train')))
    
    return min(dist[(n, 'Car')], dist[(n, 'Train')])

# Read input
n, a, b, c = map(int, input().split())
distances = []
for _ in range(n):
    distances.append(list(map(int, input().split())))

# Calculate and print the result
result = min_time(n, a, b, c, distances)
print(result)