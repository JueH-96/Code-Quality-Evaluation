import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
water = 0
last_time = 0
for _ in range(N):
    T = int(data[index])
    index += 1
    V = int(data[index])
    index += 1
    time_elapsed = T - last_time
    water = max(0, water - time_elapsed)
    water += V
    last_time = T
print(water)