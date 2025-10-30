import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
P = list(map(int, data[1:N+1]))
A = list(map(int, data[N+1:]))

# Create a mapping from P to its index
P_to_index = [0] * N
for i in range(N):
    P_to_index[P[i] - 1] = i

# Create a list to store the result
result = [0] * N

# Perform the operations to find the lexicographically smallest A
for i in range(N):
    result[i] = A[P_to_index[i]]

# Print the result
print(" ".join(map(str, result)))