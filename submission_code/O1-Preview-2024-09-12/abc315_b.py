# YOUR CODE HERE
M = int(input())
D = list(map(int, input().split()))
total_days = sum(D)
middle_day_index = (total_days +1) // 2
cum_days = 0
for i in range(M):
    cum_days += D[i]
    if cum_days >= middle_day_index:
        day_in_month = middle_day_index - (cum_days - D[i])
        print(i+1, day_in_month)
        break