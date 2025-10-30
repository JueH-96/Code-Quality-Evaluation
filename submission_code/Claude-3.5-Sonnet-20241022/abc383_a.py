# Read input
N = int(input())
events = []
for _ in range(N):
    t, v = map(int, input().split())
    events.append((t, v))

# Process each time point
water = 0
for i in range(N):
    curr_time = events[i][0]
    
    # If not first event, calculate water loss since previous event
    if i > 0:
        time_diff = curr_time - events[i-1][0]
        water = max(0, water - time_diff)
    
    # Add water at current time
    water += events[i][1]

print(water)