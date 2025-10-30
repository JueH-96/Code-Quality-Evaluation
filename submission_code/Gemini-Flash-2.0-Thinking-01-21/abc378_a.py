# YOUR CODE HERE
colors = list(map(int, input().split()))
color_counts = [0] * 5
for color in colors:
    color_counts[color] += 1

total_operations = 0
for i in range(1, 5):
    total_operations += color_counts[i] // 2

print(total_operations)