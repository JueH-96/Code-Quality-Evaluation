import math

N = int(input())

# Calculate the number of perfect powers between 1 and N
count = 0
max_base = int(math.sqrt(N))

# Count perfect squares, cubes, etc.
for base in range(2, max_base + 1):
    power = 2
    while base ** power <= N:
        count += 1
        power += 1

# Add 1 for the number 1, which is a perfect power (1^b for any b)
count += 1

print(count)