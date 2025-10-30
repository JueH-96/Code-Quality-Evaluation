n = int(input())
d = list(map(int, input().split()))
count = 0

for i in range(1, n + 1):
    s_month = str(i)
    # Check if the month is a repdigit
    if all(c == s_month[0] for c in s_month):
        days = d[i - 1]
        for j in range(1, days + 1):
            s_day = str(j)
            # Check if the day is a repdigit
            if all(c == s_day[0] for c in s_day):
                # Check if the first digits match
                if s_month[0] == s_day[0]:
                    count += 1

print(count)