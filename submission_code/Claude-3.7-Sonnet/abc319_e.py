def solve(N, X, Y, P, T, q):
    # Takahashi leaves his house at time q
    current_time = q
    
    # Walk to bus stop 1
    current_time += X
    
    # Travel through each bus stop
    for i in range(N-1):
        # Calculate next bus departure time
        # If he arrives exactly at departure time, he can take that bus
        if current_time % P[i] == 0:
            departure_time = current_time
        else:
            departure_time = (current_time // P[i] + 1) * P[i]
        
        # Update time after taking the bus to next stop
        current_time = departure_time + T[i]
    
    # Walk from bus stop N to Aoki's house
    current_time += Y
    
    return current_time

# Read input
N, X, Y = map(int, input().split())

P = []
T = []
for _ in range(N-1):
    p, t = map(int, input().split())
    P.append(p)
    T.append(t)

Q = int(input())
for _ in range(Q):
    q = int(input())
    print(solve(N, X, Y, P, T, q))