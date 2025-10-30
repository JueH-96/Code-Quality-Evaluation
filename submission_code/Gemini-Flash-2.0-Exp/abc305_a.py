n = int(input())
stations = [i * 5 for i in range(21)]
closest_station = min(stations, key=lambda x: abs(x - n))
print(closest_station)