# Read input values
A, M, L, R = map(int, input().split())

# Calculate lower and upper k values
x_low = L - A
k_low = (x_low + M - 1) // M

x_high = R - A
k_high = x_high // M

# Determine the number of Christmas trees in the interval
if k_high >= k_low:
    print(k_high - k_low + 1)
else:
    print(0)