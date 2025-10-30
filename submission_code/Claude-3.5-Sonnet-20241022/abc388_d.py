# Read input
N = int(input())
A = list(map(int, input().split()))

# Create a copy of initial stones to track changes
stones = A.copy()

# Simulate each year
for year in range(1, N+1):
    # Find who becomes adult this year (index = year-1)
    new_adult = year - 1
    
    # Count how many stones will be given
    gifts = 0
    for i in range(N):
        # If this alien became adult in previous years and has stones
        if i < year and stones[i] > 0:
            gifts += 1
            stones[i] -= 1
    
    # Add all gifts to the new adult
    stones[new_adult] += gifts

# Print final result
print(*stones)