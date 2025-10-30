n = int(input())
current_water = 0
current_time = 0
for _ in range(n):
    t, v = map(int, input().split())
    delta = t - current_time
    current_water = max(current_water - delta, 0)
    current_water += v
    current_time = t
print(current_water)