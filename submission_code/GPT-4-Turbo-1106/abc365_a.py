# Read the year from standard input
Y = int(input())

# Check the conditions for the number of days in the year
if Y % 4 != 0:
    days = 365
elif Y % 100 != 0:
    days = 366
elif Y % 400 != 0:
    days = 365
else:
    days = 366

# Print the number of days in the year
print(days)