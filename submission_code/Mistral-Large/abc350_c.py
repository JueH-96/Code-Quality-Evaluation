import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Initialize an array to keep track of the positions
pos = [0] * (N + 1)
for i in range(N):
    pos[A[i]] = i

operations = []

for i in range(1, N + 1):
    if pos[i] != i - 1:
        # Swap the element at position i-1 with the element at position pos[i]
        j = pos[i]
        operations.append((i - 1, j))

        # Update the positions
        A[i - 1], A[j] = A[j], A[i - 1]
        pos[A[i - 1]] = i - 1
        pos[A[j]] = j

# Output the result
print(len(operations))
for op in operations:
    print(op[0] + 1, op[1] + 1)