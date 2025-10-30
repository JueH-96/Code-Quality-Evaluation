def is_leap_year(year):
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    if year % 4 == 0:
        return True
    return False

def days_in_year(year):
    return 366 if is_leap_year(year) else 365

import sys
input = sys.stdin.read
year = int(input().strip())
print(days_in_year(year))