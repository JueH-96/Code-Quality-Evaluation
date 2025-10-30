# Read the input value for N
N = int(input())

# Calculate the nearest water station
# Water stations are at every 5 km, so we find the closest multiple of 5
lower = (N // 5) * 5
higher = lower + 5

# Determine which one is closer
if abs(N - lower) <= abs(N - higher):
    nearest = lower
else:
    nearest = higher

# Ensure the nearest station is within the 0-100 km range
nearest = max(0, min(nearest, 100))

# Print the result
print(nearest)