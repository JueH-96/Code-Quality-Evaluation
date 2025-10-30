# YOUR CODE HERE
Y = int(input())

if Y % 400 == 0:
    # Multiple of 400 -> 366 days
    print(366)
elif Y % 100 == 0:
    # Multiple of 100 but not 400 -> 365 days
    print(365)
elif Y % 4 == 0:
    # Multiple of 4 but not 100 -> 366 days
    print(366)
else:
    # Not a multiple of 4 -> 365 days
    print(365)