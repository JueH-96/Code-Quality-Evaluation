# Read the number of buildings
N = int(input())

# Read the heights of the buildings
heights = list(map(int, input().split()))

# Get the height of the first building
first_height = heights[0]

# Iterate through the buildings starting from the second one
for i in range(1, N):
    if heights[i] > first_height:
        print(i + 1)
        break
else:
    print(-1)