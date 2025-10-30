n = int(input())
current_water = 0
last_time = 0

for i in range(n):
    t, v = map(int, input().split())
    
    # Time passed since last addition
    time_passed = t - last_time
    
    # Water decreases by 1 per unit time (but not below 0)
    current_water = max(0, current_water - time_passed)
    
    # Add the new water
    current_water += v
    
    last_time = t

print(current_water)