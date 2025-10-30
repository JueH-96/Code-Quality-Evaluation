import sys
from collections import deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

X = list(map(int, data[2:2+M]))
A = list(map(int, data[2+M:]))

# Check if it's possible to distribute stones evenly
if sum(A) != N:
    print(-1)
    sys.exit()

# Create a deque to simulate the operations
stones = deque(A)
cells = deque([0] * N)

for i in range(M):
    cells[X[i]-1] = stones.popleft()

# Function to perform the operations
def distribute_stones():
    operations = 0
    for i in range(N-1):
        while cells[i] > 1:
            cells[i] -= 1
            cells[i+1] += 1
            operations += 1
    return operations

# Check if the distribution is possible
if all(cells):
    print(distribute_stones())
else:
    print(-1)