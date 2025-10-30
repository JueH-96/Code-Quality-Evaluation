M, D = map(int, input().split())
y, m, d = map(int, input().split())

# Add one day
d += 1

# If day exceeds D, increment month and reset day
if d > D:
    d = 1
    m += 1
    
# If month exceeds M, increment year and reset month
if m > M:
    m = 1
    y += 1

print(y, m, d)