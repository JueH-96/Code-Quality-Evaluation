# Read the input
N = int(input())
A = list(map(int, input().split()))

# Initialize the count of different kagamimochi
count = 0

# Iterate through all possible pairs of mochi
for i in range(N):
    for j in range(i+1, N):
        # Check if the size of the first mochi is at most half the size of the second mochi
        if A[i] <= A[j] / 2:
            count += 1

# Print the result
print(count)