N = int(input())

# Water stations are at 0, 5, 10, ..., 95, 100 km
# Find the closest station to N

# Try both rounding up and down to nearest 5
lower = (N // 5) * 5
upper = lower + 5 if lower < 100 else 100

# Get distances to both stations
dist_lower = abs(N - lower)
dist_upper = abs(N - upper)

# Return the station with minimum distance
if dist_lower <= dist_upper:
    print(lower)
else:
    print(upper)