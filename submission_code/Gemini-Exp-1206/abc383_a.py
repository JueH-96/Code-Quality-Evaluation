N = int(input())
water = 0
last_time = 0
for _ in range(N):
    T, V = map(int, input().split())
    water -= min(water, T - last_time)
    water += V
    last_time = T
print(water)