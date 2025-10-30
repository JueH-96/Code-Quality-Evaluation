# Read the inputs
N, M, D = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort the lists to make it easier to find pairs within the difference limit
A.sort()
B.sort()

# Initialize variables to keep track of the maximum sum and pointers for A and B
max_sum = -1
j = 0

# Iterate through each gift for Aoki
for i in range(N):
    # Move the pointer for Snuke's gifts to the right while the difference is too large
    while j < M and B[j] < A[i] - D:
        j += 1
    # Check if the current pair is valid and update the maximum sum if necessary
    if j < M and A[i] - D <= B[j] <= A[i] + D:
        max_sum = max(max_sum, A[i] + B[j])

# Print the result
print(max_sum)