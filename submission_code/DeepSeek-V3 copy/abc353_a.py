# Read input values
N = int(input())
H = list(map(int, input().split()))

# Get the height of the first building
first_height = H[0]

# Iterate through the buildings to find the first one taller than the first
for i in range(1, N):
    if H[i] > first_height:
        # Print the position (1-based index)
        print(i + 1)
        exit()

# If no such building is found, print -1
print(-1)