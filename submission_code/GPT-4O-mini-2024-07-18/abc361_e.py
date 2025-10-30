def minimum_travel_distance(N, roads):
    total_length = sum(c for _, _, c in roads)
    return total_length * 2

import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
roads = [tuple(map(int, line.split())) for line in data[1:N]]

result = minimum_travel_distance(N, roads)
print(result)