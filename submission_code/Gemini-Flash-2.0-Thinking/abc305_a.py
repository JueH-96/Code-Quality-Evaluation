n = int(input())

lower_station = (n // 5) * 5
upper_station = lower_station + 5

if upper_station > 100:
    print(lower_station)
elif n - lower_station <= upper_station - n:
    print(lower_station)
else:
    print(upper_station)