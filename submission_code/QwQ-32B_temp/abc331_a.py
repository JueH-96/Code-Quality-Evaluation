# Read input
M, D = map(int, input().split())
y, m, d = map(int, input().split())

# Calculate next day
d += 1
if d > D:
    d = 1
    m += 1
    if m > M:
        m = 1
        y += 1

# Output the result
print(y, m, d)