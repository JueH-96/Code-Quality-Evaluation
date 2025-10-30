# Read input
M, D = map(int, input().split())
y, m, d = map(int, input().split())

# Increment day
d += 1

# Check if day exceeds D
if d > D:
    d = 1
    m += 1
    # Check if month exceeds M
    if m > M:
        m = 1
        y += 1

# Output the result
print(y, m, d)