import sys
data = sys.stdin.read().split()
months_in_year = int(data[0])
days_in_month = int(data[1])
year = int(data[2])
month = int(data[3])
day = int(data[4])

if day < days_in_month:
    next_year = year
    next_month = month
    next_day = day + 1
elif month < months_in_year:
    next_year = year
    next_month = month + 1
    next_day = 1
else:
    next_year = year + 1
    next_month = 1
    next_day = 1

print(next_year, next_month, next_day)