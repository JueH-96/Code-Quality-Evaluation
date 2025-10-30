n = int(input())
water = 0
last_time = 0
for i in range(n):
    t, v = map(int, input().split())
    water -= (t - last_time) * (water > 0)
    water += v
    last_time = t
print(water)