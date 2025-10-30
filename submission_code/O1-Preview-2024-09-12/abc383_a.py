# YOUR CODE HERE
# Read N
N = int(input())
prev_time = 0
water = 0

for _ in range(N):
    T_i, V_i = map(int, input().split())
    time_elapsed = T_i - prev_time
    water = max(0, water - time_elapsed)
    water += V_i
    prev_time = T_i

print(water)