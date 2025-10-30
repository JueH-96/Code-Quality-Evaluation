# Read input
N = int(input())
A = list(map(int, input().split()))

# Initialize result array
B = []

# For each position i, find the most recent previous occurrence
for i in range(N):
    # Start from position i-1 and go backwards
    found = False
    for j in range(i-1, -1, -1):
        if A[j] == A[i]:
            B.append(j+1)  # Add 1 because positions are 1-indexed
            found = True
            break
    if not found:
        B.append(-1)

# Print result
print(*B)