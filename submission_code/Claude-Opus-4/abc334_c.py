# YOUR CODE HERE
N, K = map(int, input().split())
if K > 0:
    A = list(map(int, input().split()))
else:
    A = []

# Colors with odd number of socks (1 sock remaining)
odd_colors = sorted(A)

# Calculate minimum weirdness
total_weirdness = 0

# Pair adjacent odd colors
for i in range(0, len(odd_colors) - 1, 2):
    total_weirdness += odd_colors[i + 1] - odd_colors[i]

print(total_weirdness)