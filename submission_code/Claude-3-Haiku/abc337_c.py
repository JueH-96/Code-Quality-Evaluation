# Read the input
N = int(input())
A = [int(x) for x in input().split()]

# Initialize an empty list to store the final order
result = [0] * N

# Iterate through the input array
for i in range(N):
    if A[i] == -1:
        result[0] = i + 1
    else:
        result[A[i]] = i + 1

# Print the result
print(" ".join(map(str, result)))