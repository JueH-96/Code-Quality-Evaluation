M = int(input())
D = list(map(int, input().split()))

total_days = sum(D)
middle_day = (total_days + 1) // 2

month = 1
while middle_day > D[month - 1]:
    middle_day -= D[month - 1]
    month += 1

print(month, middle_day)