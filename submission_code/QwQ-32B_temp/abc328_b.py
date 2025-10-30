n = int(input())
d_list = list(map(int, input().split()))
count = 0

for month in range(1, n+1):
    days = d_list[month-1]
    for day in range(1, days+1):
        s = str(month) + str(day)
        if all(c == s[0] for c in s):
            count += 1

print(count)