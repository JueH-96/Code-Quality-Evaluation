N = int(input().strip())

# Calculate the nearest water station
nearest_station = (N // 5) * 5
if N % 5 > 0:
    nearest_station += 5

print(nearest_station)