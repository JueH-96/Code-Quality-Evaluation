# Read the input
k, g, m = map(int, input().split())

# Perform the operations
for _ in range(k):
    if g == 0:
        g = m
        m = 0
    elif m == 0:
        m = g
        g = 0
    else:
        if g + m <= m:
            g = 0
            m -= g
        else:
            m -= m - g
            g = m

# Print the output
print(g, m)