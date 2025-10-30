# YOUR CODE HERE
def days_in_year(Y):
    if Y % 4 == 0:
        if Y % 100 == 0:
            if Y % 400 == 0:
                return 366
            else:
                return 365
        else:
            return 366
    else:
        return 365

Y = int(input().strip())
print(days_in_year(Y))