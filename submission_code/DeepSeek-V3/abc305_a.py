# YOUR CODE HERE
N = int(input())
# Calculate the nearest water station
# Water stations are at 0, 5, 10, ..., 100 km
# Find the nearest multiple of 5 to N
lower = (N // 5) * 5
higher = lower + 5
# Determine which is closer
if abs(N - lower) <= abs(N - higher):
    print(lower)
else:
    print(higher)