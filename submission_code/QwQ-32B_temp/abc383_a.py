n = int(input())
data = []
for _ in range(n):
    t, v = map(int, input().split())
    data.append((t, v))

current_water = 0
prev_time = 0

for t, v in data:
    time_elapsed = t - prev_time
    current_water = max(current_water - time_elapsed, 0)
    current_water += v
    prev_time = t

print(current_water)