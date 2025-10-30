# Read input values
M, D = map(int, input().split())
y, m, d = map(int, input().split())

# Determine the next day
if d < D:
    next_y, next_m, next_d = y, m, d + 1
elif m < M:
    next_y, next_m, next_d = y, m + 1, 1
else:
    next_y, next_m, next_d = y + 1, 1, 1

# Output the result
print(next_y, next_m, next_d)