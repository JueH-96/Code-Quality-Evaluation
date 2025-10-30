# Read the total number of months in a year (M) and days in a month (D)
M, D = map(int, input().split())

# Read the current year (y), month (m), and day (d)
y, m, d = map(int, input().split())

# Initialize variables for the next day, month, and year
next_y = y
next_m = m
next_d = d + 1 # Assume the day simply increments

# Check if the day has overflowed (i.e., it's past the last day of the month)
if next_d > D:
    next_d = 1      # Reset day to 1
    next_m += 1     # Increment month

    # Check if the month has overflowed (i.e., it's past the last month of the year)
    if next_m > M:
        next_m = 1  # Reset month to 1
        next_y += 1 # Increment year

# Print the calculated next year, month, and day, separated by spaces
print(next_y, next_m, next_d)