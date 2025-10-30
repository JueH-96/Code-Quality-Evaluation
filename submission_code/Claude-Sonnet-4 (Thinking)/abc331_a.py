# YOUR CODE HERE
M, D = map(int, input().split())
y, m, d = map(int, input().split())

if d < D:
    # Not the last day of the month
    print(y, m, d + 1)
elif m < M:
    # Last day of month, but not last month of year
    print(y, m + 1, 1)
else:
    # Last day of last month of year
    print(y + 1, 1, 1)