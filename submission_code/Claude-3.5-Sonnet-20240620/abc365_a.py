# YOUR CODE HERE
def days_in_year(year):
    if year % 400 == 0:
        return 366
    elif year % 100 == 0:
        return 365
    elif year % 4 == 0:
        return 366
    else:
        return 365

Y = int(input())
print(days_in_year(Y))