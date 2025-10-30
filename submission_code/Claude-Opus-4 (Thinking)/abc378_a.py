# YOUR CODE HERE
balls = list(map(int, input().split()))

# Count frequency of each color
count = {}
for ball in balls:
    count[ball] = count.get(ball, 0) + 1

# Calculate maximum operations
operations = 0
for freq in count.values():
    operations += freq // 2

print(operations)