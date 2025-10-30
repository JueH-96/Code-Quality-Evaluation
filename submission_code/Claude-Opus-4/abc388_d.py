# YOUR CODE HERE
n = int(input())
stones = list(map(int, input().split()))

# Simulate year by year
for year in range(1, n + 1):
    # The alien at index (year-1) becomes an adult after 'year' years
    new_adult_idx = year - 1
    
    # Count gifts for the new adult
    gifts = 0
    
    # All aliens with index < new_adult_idx are already adults
    for i in range(new_adult_idx):
        if stones[i] > 0:
            gifts += 1
            stones[i] -= 1
    
    # The new adult receives all the gifts
    stones[new_adult_idx] += gifts

print(' '.join(map(str, stones)))