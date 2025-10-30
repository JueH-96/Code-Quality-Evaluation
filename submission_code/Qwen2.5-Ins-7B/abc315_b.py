# YOUR CODE HERE
M = int(input())
days = list(map(int, input().split()))
total_days = sum(days)
middle_day = (total_days + 1) // 2
current_day = 1
month = 1

for i in range(M):
    if middle_day <= current_day + days[i] - 1:
        break
    current_day += days[i]
    month += 1

print(month, middle_day - current_day + 1)