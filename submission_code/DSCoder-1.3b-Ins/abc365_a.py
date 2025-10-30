# YOUR CODE HERE

def count_days_in_year(year):
    if year % 4 != 0:
        return 365
    elif year % 100 != 0:
        return 366
    elif year % 400 != 0:
        return 365
    else:
        return 366

year = int(input())
print(count_days_in_year(year))