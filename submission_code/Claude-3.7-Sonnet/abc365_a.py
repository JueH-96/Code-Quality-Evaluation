# YOUR CODE HERE
Y = int(input())

if Y % 400 == 0:
    print(366)  # If divisible by 400, it's a leap year (366 days)
elif Y % 100 == 0:
    print(365)  # If divisible by 100 but not 400, not a leap year (365 days)
elif Y % 4 == 0:
    print(366)  # If divisible by 4 but not 100, it's a leap year (366 days)
else:
    print(365)  # If not divisible by 4, not a leap year (365 days)