# Read input
N = int(input())
A = list(map(int, input().split()))

# Count valid kagamimochi combinations
count = 0

# Try all possible pairs of mochi
for i in range(N):
    for j in range(N):
        # Skip if trying to stack same mochi
        if i == j:
            continue
            
        # Check if mochi i can be placed on top of mochi j
        if A[i] * 2 <= A[j]:
            count += 1

# Output result
print(count)