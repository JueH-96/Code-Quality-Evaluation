import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
P = list(map(int, data[2:]))

# Create a list to store the result of each index after K operations
result = [0] * N

# Create a list to store the index of each value in P
index = [0] * N
for i in range(N):
    index[P[i] - 1] = i

# Create a list to store the result of each index after one operation
next_index = [0] * N
for i in range(N):
    next_index[i] = index[P[i] - 1]

# Use the property that K operations are equivalent to K % N operations
K %= N

# Use the property that K operations are equivalent to K operations on the next_index list
for _ in range(K):
    next_index = [index[next_index[i]] for i in range(N)]

# Create the result list
for i in range(N):
    result[next_index[i]] = i + 1

# Print the result
print(" ".join(map(str, result)))