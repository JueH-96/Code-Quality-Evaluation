# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])

functions = []
for i in range(N):
    A = int(data[2 + 2 * i])
    B = int(data[3 + 2 * i])
    functions.append((A, B))

max_value = 0
for p in range(1, 1 << N):
    if bin(p).count('1') != K:
        continue
    x = 1
    for i in range(N):
        if p & (1 << i):
            A, B = functions[i]
            x = A * x + B
    max_value = max(max_value, x)

print(max_value)