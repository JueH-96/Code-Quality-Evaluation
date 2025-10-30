A, M, L, R = map(int, input().split())

# Calculate ceil((L - A) / M)
ceil_div = -(- (L - A) // M)

# Calculate floor((R - A) / M)
floor_div = (R - A) // M

# Calculate the number of valid k
count = floor_div - ceil_div + 1

# Ensure count is not negative
if count < 0:
    count = 0

print(count)