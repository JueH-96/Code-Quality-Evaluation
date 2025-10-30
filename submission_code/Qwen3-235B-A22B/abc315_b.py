# Read input
M = int(input())
D = list(map(int, input().split()))

# Calculate the total number of days and the target day
total_days = sum(D)
target_day = (total_days + 1) // 2

sum_days = 0

# Iterate through each month to find the target day
for i in range(M):
    current_month = i + 1
    days = D[i]
    if sum_days < target_day <= sum_days + days:
        a = current_month
        b = target_day - sum_days
        print(a, b)
        break
    else:
        sum_days += days