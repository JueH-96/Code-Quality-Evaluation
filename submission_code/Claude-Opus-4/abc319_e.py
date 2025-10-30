def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

# Read input
N, X, Y = map(int, input().split())
buses = []
for _ in range(N-1):
    P, T = map(int, input().split())
    buses.append((P, T))

# Calculate LCM of all P values
period = 1
for P, _ in buses:
    period = lcm(period, P)

# Precompute minimum time from bus stop 1 to Aoki's house for each starting time mod period
min_time = [float('inf')] * period

for start_mod in range(period):
    # Simulate journey starting at bus stop 1 at time start_mod
    current_time = start_mod
    
    # Go through each bus stop
    for i in range(N-1):
        P, T = buses[i]
        # Wait for next bus
        wait_time = (P - current_time % P) % P
        current_time += wait_time + T
    
    # Add time to walk from bus stop N to Aoki's house
    total_time = current_time + Y - start_mod
    min_time[start_mod] = total_time

# Process queries
Q = int(input())
for _ in range(Q):
    q = int(input())
    
    # Time to reach bus stop 1
    arrival_at_stop1 = q + X
    
    # Find the minimum time considering the periodic pattern
    best_time = float('inf')
    
    # Check the current period and a few subsequent ones
    for k in range(3):  # Check enough periods to ensure we find the minimum
        base_time = arrival_at_stop1 + k * period
        mod_time = base_time % period
        
        # Time from bus stop 1 to Aoki's house if we start at base_time
        time_from_stop1 = min_time[mod_time] + (base_time - mod_time)
        
        # Total time from leaving house
        total_time = q + X + time_from_stop1
        
        best_time = min(best_time, total_time)
        
        # Early termination if we've found a good solution
        if time_from_stop1 == min_time[mod_time]:
            break
    
    print(best_time)