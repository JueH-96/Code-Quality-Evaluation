# YOUR CODE HERE
n = int(input())
water = 0
prev_time = 0

for i in range(n):
    t, v = map(int, input().split())
    
    # Calculate water lost due to leak since last addition
    if i > 0:
        time_passed = t - prev_time
        water = max(0, water - time_passed)
    
    # Add new water
    water += v
    prev_time = t

print(water)