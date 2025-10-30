# YOUR CODE HERE
M = int(input())
D = list(map(int, input().split()))

total_days = sum(D)
mid_day = (total_days + 1) // 2

day_count = 0
for i in range(M):
    day_count += D[i]
    if day_count >= mid_day:
        print(i + 1, mid_day - (day_count - D[i]))
        break