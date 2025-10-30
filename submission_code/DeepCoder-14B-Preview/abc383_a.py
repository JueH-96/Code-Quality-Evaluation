n = int(input())
additions = []
for _ in range(n):
    t, v = map(int, input().split())
    additions.append((t, v))

current_water = 0
current_time = 0

for t, v in additions:
    delta = t - current_time
    current_water = max(0, current_water - delta)
    current_water += v
    current_time = t

print(current_water)