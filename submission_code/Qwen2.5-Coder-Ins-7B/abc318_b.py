# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
sheets = []

for i in range(N):
    A = int(data[4*i+1])
    B = int(data[4*i+2])
    C = int(data[4*i+3])
    D = int(data[4*i+4])
    sheets.append((A, B, C, D))

# Initialize a 2D array to keep track of covered cells
covered = [[False] * 101 for _ in range(101)]

# Mark the cells covered by each sheet
for A, B, C, D in sheets:
    for x in range(A, B):
        for y in range(C, D):
            covered[x][y] = True

# Count the number of covered cells
S = sum(sum(row) for row in covered)

# Print the result
print(S)