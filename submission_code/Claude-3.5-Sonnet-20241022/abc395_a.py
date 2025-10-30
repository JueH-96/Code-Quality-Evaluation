# Read N from stdin
N = int(input())

# Read the sequence A from stdin
A = list(map(int, input().split()))

# Check if sequence is strictly increasing
is_strictly_increasing = True
for i in range(N-1):
    if A[i] >= A[i+1]:
        is_strictly_increasing = False
        break

# Print result (case doesn't matter as per problem statement)
print("Yes" if is_strictly_increasing else "No")