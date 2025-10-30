# YOUR CODE HERE
H = int(input())

# Initialize the height of the plant and the day
plant_height = 0
day = 0

# Loop until the plant's height is strictly greater than Takahashi's height
while plant_height <= H:
    plant_height += 2 ** day
    day += 1

# Print the first day when the plant's height is strictly greater than Takahashi's height
print(day)