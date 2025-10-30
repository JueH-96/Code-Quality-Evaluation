# YOUR CODE HERE

# Read the input
Y = int(input())

# Check if the year is a leap year
if Y % 400 == 0:
    print(366)
elif Y % 100 == 0:
    print(365)
elif Y % 4 == 0:
    print(366)
else:
    print(365)