# YOUR CODE HERE
n = int(input())
current_water = 0
last_time = 0

for _ in range(n):
    t, v = map(int, input().split())
    
    # Calculate water lost due to leak since last time
    time_elapsed = t - last_time
    current_water = max(0, current_water - time_elapsed)
    
    # Add new water
    current_water += v
    
    # Update last time
    last_time = t

print(current_water)