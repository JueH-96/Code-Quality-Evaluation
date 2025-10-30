# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
C = int(data[2])
K = int(data[3])
A = list(map(int, data[4:]))

# Calculate the minimum values for each k
min_values = []
for k in range(K):
    min_val = min((C * k + a) % M for a in A)
    min_values.append(min_val)

# Sum up the minimum values
result = sum(min_values)

# Print the result
print(result)