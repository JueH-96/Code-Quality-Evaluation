n = int(input())
current_time = 0
current_water = 0
for _ in range(n):
    t, v = map(int, input().split())
    current_water -= (t - current_time)
    if current_water < 0:
        current_water = 0
    current_water += v
    current_time = t
print(current_water)