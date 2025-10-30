# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

# Process each year
for year in range(1, N+1):
    # Alien with index (year-1) becomes adult
    new_adult_idx = year - 1
    
    # Count gifts from existing adults
    gifts = 0
    for adult_idx in range(new_adult_idx):
        if A[adult_idx] > 0:
            A[adult_idx] -= 1
            gifts += 1
    
    # Give gifts to new adult
    A[new_adult_idx] += gifts

print(*A)