# Read M and D
M, D = map(int, input().split())

# Read y, m, d
y, m, d = map(int, input().split())

# Determine the next day
if d < D:
    d += 1
elif d == D:
    d = 1
    if m < M:
        m += 1
    else:
        m = 1
        y += 1

# Print the result
print(y, m, d)