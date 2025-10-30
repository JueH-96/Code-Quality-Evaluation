# YOUR CODE HERE
import sys

# Read input
H = int(sys.stdin.read().strip())

# Initialize variables
day = 0
plant_height = 0

# Loop until the plant's height is greater than Takahashi's height
while plant_height <= H:
    day += 1
    plant_height += 2 ** (day - 1)

# Print the first day the plant is taller than Takahashi
print(day)