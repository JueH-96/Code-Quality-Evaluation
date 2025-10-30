# YOUR CODE HERE
import sys

# Read input from stdin
Y = int(sys.stdin.read().strip())

# Determine the number of days in the year
if Y % 400 == 0:
    days = 366
elif Y % 100 == 0:
    days = 365
elif Y % 4 == 0:
    days = 366
else:
    days = 365

# Print the result
print(days)