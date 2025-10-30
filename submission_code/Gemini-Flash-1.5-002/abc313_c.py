# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

sum_a = sum(a)
avg = sum_a / n
avg_floor = int(avg // 1)
count_floor = 0
count_ceil = 0

for x in a:
    if x > avg_floor:
        count_ceil += x - avg_floor
    else:
        count_floor += avg_floor - x

print(max(count_floor, count_ceil))