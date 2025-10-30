def next_bus_time(current_time, period):
    return ((current_time + period - 1) // period) * period

def solve():
    N, X, Y = map(int, input().split())
    buses = []
    for _ in range(N-1):
        p, t = map(int, input().split())
        buses.append((p, t))
    
    Q = int(input())
    queries = [int(input()) for _ in range(Q)]
    
    for start_time in queries:
        # Time to reach first bus stop
        current_time = start_time + X
        current_pos = 1
        
        # Take buses from stop 1 to N
        for period, travel_time in buses:
            # Wait for next available bus
            bus_time = next_bus_time(current_time, period)
            # Update time after taking bus
            current_time = bus_time + travel_time
            current_pos += 1
            
        # Walk from last stop to Aoki's house
        arrival_time = current_time + Y
        print(arrival_time)

solve()