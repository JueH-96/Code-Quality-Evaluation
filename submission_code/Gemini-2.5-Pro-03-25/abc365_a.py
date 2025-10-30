# YOUR CODE HERE
import sys

# Read the input year Y
Y = int(sys.stdin.readline())

# Determine the number of days based on the Gregorian calendar rules
# Check the rules in order of precedence (most specific first)

# Rule 4: If Y is a multiple of 400, then 366 days.
if Y % 400 == 0:
    days = 366
# Rule 3: If Y is a multiple of 100 but not a multiple of 400, then 365 days.
# (Since we already checked for multiples of 400, this elif handles multiples of 100 that are *not* multiples of 400)
elif Y % 100 == 0:
    days = 365
# Rule 2: If Y is a multiple of 4 but not a multiple of 100, then 366 days.
# (Since we already checked for multiples of 100, this elif handles multiples of 4 that are *not* multiples of 100)
elif Y % 4 == 0:
    days = 366
# Rule 1: If Y is not a multiple of 4, then 365 days.
# (This is the remaining case)
else:
    days = 365

# Print the result
print(days)