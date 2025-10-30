import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
C = int(data[2])
K = int(data[3])
A = list(map(int, data[4:]))

# Calculate the minimum value for each k
min_values = []
for k in range(K):
    min_value = float('inf')
    for i in range(N):
        value = (C * k + A[i]) % M
        if value < min_value:
            min_value = value
    min_values.append(min_value)

# Sum the minimum values
result = sum(min_values)

# Print the result
print(result)