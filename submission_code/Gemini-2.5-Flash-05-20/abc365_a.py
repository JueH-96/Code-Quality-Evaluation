# YOUR CODE HERE
Y = int(input())

# Check for leap year conditions based on Gregorian calendar rules:
# A year is a leap year (366 days) if:
# 1. It is divisible by 4, BUT not by 100
# OR
# 2. It is divisible by 400
if (Y % 4 == 0 and Y % 100 != 0) or (Y % 400 == 0):
    print(366)
else:
    print(365)