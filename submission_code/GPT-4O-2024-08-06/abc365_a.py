# YOUR CODE HERE
def number_of_days_in_year(Y):
    if Y % 4 != 0:
        return 365
    elif Y % 100 != 0:
        return 366
    elif Y % 400 != 0:
        return 365
    else:
        return 366

# Read input from standard input
import sys
input = sys.stdin.read
Y = int(input().strip())

# Calculate and print the number of days in the year Y
print(number_of_days_in_year(Y))