# Read input
N = int(input())
P = list(map(int, input().split()))

# Initialize empty array
A = []

# Process each number from 1 to N
for i in range(1, N + 1):
    # Insert number i at position P[i-1]-1
    # We use P[i-1] because P is 0-based indexed but input is 1-based
    A.insert(P[i-1]-1, i)

# Print result
print(*A)