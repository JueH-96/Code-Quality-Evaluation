# Read the input year
Y = int(input())

# Determine the number of days based on the Gregorian calendar rules
if Y % 4 != 0:
    print(365)
elif Y % 100 != 0:
    print(366)
elif Y % 400 != 0:
    print(365)
else:
    print(366)