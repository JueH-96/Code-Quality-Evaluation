import math
import sys

# Read all input data
data = sys.stdin.read().split()
index = 0

# Read N
N = int(data[index])
index += 1

# Create a list of positions including start and end points
pos = [(0, 0)]  # Start at origin
for _ in range(N):
    X = int(data[index])
    Y = int(data[index + 1])
    pos.append((X, Y))  # Add each point
    index += 2
pos.append((0, 0))  # Return to origin

# Calculate the total sum of distances
total_sum = 0.0
for i in range(len(pos) - 1):
    x1, y1 = pos[i]
    x2, y2 = pos[i + 1]
    dx = x1 - x2
    dy = y1 - y2
    dist = math.sqrt(dx * dx + dy * dy)
    total_sum += dist

# Print the result with high precision
print("{:.15f}".format(total_sum))