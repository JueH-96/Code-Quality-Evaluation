# YOUR CODE HERE
N, X, Y = map(int, input().split())
buses = []
for _ in range(N - 1):
    P, T = map(int, input().split())
    buses.append((P, T))

Q = int(input())

for _ in range(Q):
    q = int(input())
    
    # Start from house at time q
    current_time = q
    
    # Walk to bus stop 1
    current_time += X
    
    # Take buses from stop 1 to stop N
    for i in range(N - 1):
        P, T = buses[i]
        # Find next bus departure
        next_departure = ((current_time + P - 1) // P) * P
        # Arrive at next stop
        current_time = next_departure + T
    
    # Walk from bus stop N to Aoki's house
    current_time += Y
    
    print(current_time)