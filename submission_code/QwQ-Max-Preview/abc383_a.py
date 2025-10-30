n = int(input())
events = []
for _ in range(n):
    t, v = map(int, input().split())
    events.append((t, v))

current_time = 0
current_water = 0

for t, v in events:
    delta = t - current_time
    if current_water > 0:
        current_water = max(current_water - delta, 0)
    current_water += v
    current_time = t

print(current_water)