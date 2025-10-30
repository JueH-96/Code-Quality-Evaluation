n = int(input())
current_water = 0
prev_time = 0

for _ in range(n):
    t, v = map(int, input().split())
    time_diff = t - prev_time
    current_water = max(0, current_water - time_diff)
    current_water += v
    prev_time = t

print(current_water)