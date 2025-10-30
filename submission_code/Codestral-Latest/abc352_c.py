import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = [0] * N
B = [0] * N

for i in range(N):
    A[i] = int(data[2 * i + 1])
    B[i] = int(data[2 * i + 2])

# Sort giants by A_i in ascending order and B_i in descending order
giants = sorted(zip(A, B), key=lambda x: (x[0], -x[1]))

# Calculate the maximum possible height
max_height = 0
current_height = 0

for a, b in giants:
    current_height += a
    max_height = max(max_height, current_height + b)

print(max_height)