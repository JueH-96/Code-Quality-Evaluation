def get_repdigit_digit(n):
    s = str(n)
    first = s[0]
    for c in s:
        if c != first:
            return None
    return first

n = int(input())
d_list = list(map(int, input().split()))
count = 0

for month in range(1, n + 1):
    d_month = get_repdigit_digit(month)
    if d_month is None:
        continue
    max_day = d_list[month - 1]
    for day in range(1, max_day + 1):
        d_day = get_repdigit_digit(day)
        if d_day is not None and d_day == d_month:
            count += 1

print(count)