import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Initialize the result
result = 0

# Calculate the prefix XOR array
prefix_xor = [0] * (N + 1)
for i in range(1, N + 1):
    prefix_xor[i] = prefix_xor[i - 1] ^ A[i - 1]

# Calculate the sum of XORs for all subarrays
for i in range(1, N):
    for j in range(i + 1, N + 1):
        result += prefix_xor[j] ^ prefix_xor[i - 1]

print(result)