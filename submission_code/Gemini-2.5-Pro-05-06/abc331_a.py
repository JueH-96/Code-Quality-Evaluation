# Read M, D (months per year, days per month)
M, D = map(int, input().split())

# Read current date y, m, d
y, m, d = map(int, input().split())

# Initialize the next date variables with the current date
next_y = y
next_m = m
next_d = d

# Increment the day
next_d += 1

# Check if the day has rolled over to the next month
if next_d > D:
    next_d = 1  # Reset day to 1st
    next_m += 1 # Increment month
    
    # Check if the month has rolled over to the next year
    if next_m > M:
        next_m = 1  # Reset month to 1st
        next_y += 1 # Increment year

# Print the calculated next date
print(next_y, next_m, next_d)