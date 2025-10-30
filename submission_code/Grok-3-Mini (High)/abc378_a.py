import sys

# Read input from stdin
data = sys.stdin.read().strip()
numbers = list(map(int, data.split()))

# Count the frequency of each color
count = [0] * 5  # Index 0 is unused, indices 1-4 for colors

for num in numbers:
    count[num] += 1

# Calculate the maximum number of pairs
pairs = 0
for i in range(1, 5):
    pairs += count[i] // 2

# Output the result
print(pairs)