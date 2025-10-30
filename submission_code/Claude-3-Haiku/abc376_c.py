import sys

# Read input
N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))
B = list(map(int, sys.stdin.readline().strip().split()))

# Find the minimum size of the box that can fit all toys
min_box_size = max(A)
for i in range(N-1):
    min_box_size = max(min_box_size, B[i])

# Check if it's possible to place all toys in separate boxes
for i in range(N):
    if A[i] > min_box_size:
        print(-1)
        sys.exit()

# Print the minimum size of the box
print(min_box_size)