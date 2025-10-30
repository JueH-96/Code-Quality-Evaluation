# Read the year from standard input
Y = int(input())

# Apply the Gregorian calendar rules to determine the number of days
if Y % 400 == 0:
    # If Y is a multiple of 400, it's a leap year.
    days = 366
elif Y % 100 == 0:
    # If Y is a multiple of 100 but not 400, it's a common year.
    days = 365
elif Y % 4 == 0:
    # If Y is a multiple of 4 but not 100, it's a leap year.
    days = 366
else:
    # If Y is not a multiple of 4, it's a common year.
    days = 365

# Print the number of days
print(days)