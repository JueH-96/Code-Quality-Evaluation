# Read the input values
M, D = map(int, input().split())
y, m, d = map(int, input().split())

if d < D:
    d += 1
else:
    if m < M:
        m += 1
        d = 1
    else:
        y += 1
        m = 1
        d = 1

# Output the result
print(y, m, d)