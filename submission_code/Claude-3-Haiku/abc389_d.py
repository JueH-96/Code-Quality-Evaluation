import math

# Read the input from stdin
R = int(input())

# Calculate the number of squares completely contained in the circle
count = 0
for i in range(int(-R), int(R) + 1):
    for j in range(int(-R), int(R) + 1):
        if math.sqrt(i**2 + j**2) <= R:
            count += 1

# Write the answer to stdout
print(count)