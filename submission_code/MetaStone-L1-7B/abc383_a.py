n = int(input())
times = []
volumes = []
for _ in range(n):
    t, v = map(int, input().split())
    times.append(t)
    volumes.append(v)

current_water = 0
prev_t = 0
for i in range(n):
    t = times[i]
    v = volumes[i]
    delta_t = t - prev_t
    current_water = max(current_water - delta_t, 0)
    current_water += v
    prev_t = t

print(current_water)