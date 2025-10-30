n = int(input())
timeline = []
for _ in range(n):
    t, v = map(int, input().split())
    timeline.append((t, v))

current_water = 0
last_time = 0

for t, v in timeline:
    time_diff = t - last_time
    current_water = max(current_water - time_diff, 0)
    current_water += v
    last_time = t

print(current_water)