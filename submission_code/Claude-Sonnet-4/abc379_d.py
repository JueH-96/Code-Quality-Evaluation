import bisect

Q = int(input())
plant_times = []  # Times when plants were planted
current_time = 0

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Plant a new plant at current time
        plant_times.append(current_time)
    
    elif query[0] == 2:
        # Wait T days
        T = query[1]
        current_time += T
    
    else:  # query[0] == 3
        # Harvest plants with height >= H
        H = query[1]
        # A plant planted at time t has height (current_time - t)
        # We want current_time - t >= H, so t <= current_time - H
        threshold_time = current_time - H
        
        # Count plants planted at or before threshold_time
        # Use binary search to find the rightmost position where plant_time <= threshold_time
        harvest_count = bisect.bisect_right(plant_times, threshold_time)
        
        print(harvest_count)
        
        # Remove harvested plants (those planted at or before threshold_time)
        plant_times = plant_times[harvest_count:]