import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
times = []
volumes = []

for i in range(1, N + 1):
    times.append(int(data[2 * i - 1]))
    volumes.append(int(data[2 * i]))

current_water = 0
previous_time = 0

for i in range(N):
    time_passed = times[i] - previous_time
    current_water = max(0, current_water - time_passed)
    current_water += volumes[i]
    previous_time = times[i]

print(current_water)