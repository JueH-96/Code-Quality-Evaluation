# Read the input
M, D = map(int, input().split())
y, m, d = map(int, input().split())

# Calculate the next day
d += 1
if d > D:
    d = 1
    m += 1
    if m > M:
        m = 1
        y += 1

# Print the output
print(y, m, d)