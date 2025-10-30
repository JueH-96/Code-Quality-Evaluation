import sys

# Read all input and split into a list
data = sys.stdin.read().split()
index = 0

# Read N
N = int(data[index])
index += 1

# Read A array, using 1-based indexing by ignoring index 0
A = [0] * (N + 1)
for i in range(1, N + 1):
    A[i] = int(data[index])
    index += 1

# Find the front person (where A[i] == -1)
for i in range(1, N + 1):
    if A[i] == -1:
        front = i
        break

# Build the successor array (succ[i] is the person directly behind i)
succ = [-1] * (N + 1)
for j in range(1, N + 1):
    if A[j] != -1:
        succ[A[j]] = j

# Traverse from front to back and store the order
current = front
order = []
while current != -1:
    order.append(current)
    current = succ[current]

# Output the order separated by spaces
print(' '.join(map(str, order)))