# Read the input year from standard input
Y = int(input())

# Determine if it's a leap year
is_leap = (Y % 4 == 0 and Y % 100 != 0) or (Y % 400 == 0)

# Output the number of days
if is_leap:
    print(366)
else:
    print(365)