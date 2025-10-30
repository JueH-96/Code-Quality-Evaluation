# Read the number of buildings
N = int(input())

# Read the heights of the buildings
heights = list(map(int, input().split()))

# Get the height of the first building
first_height = heights[0]

# Find the leftmost building taller than the first one
for i in range(1, N):
    if heights[i] > first_height:
        print(i + 1)  # Adding 1 to convert from 0-indexed to 1-indexed
        break
else:  # This else corresponds to the for loop. It runs if the loop completes without a break.
    print(-1)