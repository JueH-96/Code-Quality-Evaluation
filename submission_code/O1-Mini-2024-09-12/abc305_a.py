N = int(input())
stations = list(range(0, 101, 5))
nearest_station = min(stations, key=lambda x: abs(x - N))
print(nearest_station)