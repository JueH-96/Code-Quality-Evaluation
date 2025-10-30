# Read input values
M, D = map(int, input().split())
y, m, d = map(int, input().split())

# Determine the next day
if d < D:
    next_y, next_m, next_day = y, m, d + 1
else:
    if m < M:
        next_y, next_m, next_day = y, m + 1, 1
    else:
        next_y, next_m, next_day = y + 1, 1, 1

# Output the result
print(next_y, next_m, next_day)