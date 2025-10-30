# Read input
n = int(input())
stones = list(map(int, input().split()))

# Keep track of which aliens are adults
is_adult = [False] * n

for year in range(1, n+1):
    # The alien becoming an adult this year (0-indexed)
    new_adult = year - 1
    is_adult[new_adult] = True
    
    # Have all adult aliens with at least one stone give a stone to the new adult
    for i in range(n):
        if is_adult[i] and i != new_adult and stones[i] > 0:
            stones[i] -= 1
            stones[new_adult] += 1

# Print the output
print(" ".join(map(str, stones)))