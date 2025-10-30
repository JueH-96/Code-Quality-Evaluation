# YOUR CODE HERE
M = int(input())
D = list(map(int, input().split()))

total_days = sum(D)
middle_day = (total_days + 1) // 2

cumulative_days = 0
for i in range(M):
    cumulative_days += D[i]
    if cumulative_days >= middle_day:
        month = i + 1
        day = middle_day - (cumulative_days - D[i])
        print(month, day)
        break