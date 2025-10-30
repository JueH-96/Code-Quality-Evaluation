# YOUR CODE HERE
n = int(input())
water = 0
last_time = 0

for i in range(n):
    t, v = map(int, input().split())
    # Calculate water lost due to leak
    time_elapsed = t - last_time
    water = max(0, water - time_elapsed)
    # Add new water
    water += v
    last_time = t

print(water)