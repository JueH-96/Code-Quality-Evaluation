# Read M and D from the first line
M, D = map(int, input().split())
# Read y, m, and d from the second line
y, m, d = map(int, input().split())

# Calculate the next date
if d < D:
    # If the current day is not the last day of the month
    d += 1
else:
    # If the current day is the last day of the month
    d = 1
    if m < M:
        # If the current month is not the last month of the year
        m += 1
    else:
        # If the current month is the last month of the year
        m = 1
        y += 1

# Print the next date
print(y, m, d)