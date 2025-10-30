import math

# Read input
N, X, Y = map(int, input().split())
buses = []
for _ in range(N-1):
    P, T = map(int, input().split())
    buses.append((P, T))

Q = int(input())
queries = []
for _ in range(Q):
    queries.append(int(input()))

# Process each query
for q in queries:
    current_time = q
    
    # Walk to bus stop 1
    current_time += X
    
    # Take buses from stop 1 to stop N
    for i in range(N-1):
        P, T = buses[i]
        
        # Find the next departure time that is >= current_time
        # Next departure is the smallest multiple of P that is >= current_time
        if current_time % P == 0:
            departure_time = current_time
        else:
            departure_time = ((current_time // P) + 1) * P
        
        # Take the bus
        current_time = departure_time + T
    
    # Walk from bus stop N to Aoki's house
    current_time += Y
    
    print(current_time)