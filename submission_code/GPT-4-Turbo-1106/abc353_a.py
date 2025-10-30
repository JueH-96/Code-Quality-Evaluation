# Read the number of buildings
N = int(input())

# Read the heights of the buildings
heights = list(map(int, input().split()))

# Find the position of the leftmost building taller than the first one
position = -1
for i in range(1, N):
    if heights[i] > heights[0]:
        position = i + 1
        break

# Print the result
print(position)