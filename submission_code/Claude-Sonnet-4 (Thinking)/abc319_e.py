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
    
    # Start at time q, walk to bus stop 1
    current_time = q + X
    
    # Go through each bus stop
    for i in range(N-1):
        # Find next departure time that is >= current_time
        next_departure = ((current_time + P[i] - 1) // P[i]) * P[i]
        
        # Take the bus
        current_time = next_departure + T[i]
    
    # Walk to Aoki's house
    current_time += Y
    
    print(current_time)