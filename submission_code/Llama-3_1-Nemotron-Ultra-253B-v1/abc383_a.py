n = int(input())
additions = [tuple(map(int, input().split())) for _ in range(n)]

current_water = 0
previous_time = 0

for t, v in additions:
    delta = t - previous_time
    current_water = max(current_water - delta, 0)
    current_water += v
    previous_time = t

print(current_water)