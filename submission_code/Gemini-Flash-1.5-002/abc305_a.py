# YOUR CODE HERE
n = int(input())
nearest_station = (n // 5) * 5
if n - nearest_station <= 5 - (n - nearest_station):
    print(nearest_station)
else:
    print(nearest_station + 5)