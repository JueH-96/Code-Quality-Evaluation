# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
times = []
volumes = []

for i in range(N):
    times.append(int(data[2 * i + 1]))
    volumes.append(int(data[2 * i + 2]))

current_time = 0
current_water = 0

for i in range(N):
    time = times[i]
    volume = volumes[i]

    # Calculate the water loss from the current time to the next addition time
    water_loss = time - current_time
    current_water = max(0, current_water - water_loss)

    # Add the new water
    current_water += volume

    # Update the current time
    current_time = time

# Calculate the water loss from the last addition time to the end
water_loss = current_time - times[-1]
current_water = max(0, current_water - water_loss)

print(current_water)