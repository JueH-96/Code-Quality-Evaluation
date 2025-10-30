# YOUR CODE HERE
m = int(input())
d = list(map(int, input().split()))

middle_day = (sum(d) + 1) // 2
current_day = 0
month = 0

for i in range(m):
    if current_day + d[i] < middle_day:
        current_day += d[i]
        month += 1
    else:
        day = middle_day - current_day
        month += 1
        print(month, day)
        break