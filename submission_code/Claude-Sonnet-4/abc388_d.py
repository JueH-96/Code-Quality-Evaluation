# YOUR CODE HERE
n = int(input())
stones = list(map(int, input().split()))

# Simulate year by year
for year in range(1, n + 1):
    # Alien at index (year-1) becomes adult
    new_adult_idx = year - 1
    
    # Count stones given by existing adults
    stones_received = 0
    for i in range(new_adult_idx):
        if stones[i] > 0:
            stones[i] -= 1  # Give away 1 stone
            stones_received += 1
    
    # New adult receives the stones
    stones[new_adult_idx] += stones_received

print(*stones)