N = int(input())

# Calculate the nearest multiple of 5
nearest_station = round(N / 5) * 5

# If rounding gives us a station beyond 100, we should use 100
if nearest_station > 100:
    nearest_station = 100

print(nearest_station)