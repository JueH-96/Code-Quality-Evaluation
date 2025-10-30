# Read input
N = int(input())
S = list(map(int, input().split()))

# Sort the array for easier processing
S.sort()

# Create a set for O(1) lookup
S_set = set(S)

# Count fine triplets
count = 0

# For each possible middle element B
for i in range(N):
    B = S[i]
    
    # For each possible first element A (must be less than B)
    for j in range(i):
        A = S[j]
        
        # Calculate what C should be to make it a fine triplet
        # B - A = C - B
        # C = 2B - A
        C = 2 * B - A
        
        # Check if C exists in S and is greater than B
        if C in S_set and C > B:
            count += 1

# Output result
print(count)