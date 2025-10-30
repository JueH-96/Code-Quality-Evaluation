# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Initialize the result
result = 0

# Iterate over all possible starting points
for i in range(N):
    xor_value = 0
    # Iterate over all possible ending points starting from i
    for j in range(i, N):
        xor_value ^= A[j]
        result += xor_value

# Print the result
print(result)