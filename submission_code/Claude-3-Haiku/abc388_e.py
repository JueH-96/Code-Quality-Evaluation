# Read the input
N = int(input())
A = list(map(int, input().split()))

# Sort the mochi in ascending order
A.sort()

# Initialize the count of kagamimochi
K = 0

# Iterate through the mochi
for i in range(N):
    for j in range(i+1, N):
        # Check if the current pair of mochi can form a kagamimochi
        if A[j] <= 2 * A[i]:
            K += 1
        else:
            break

# Print the answer
print(K)