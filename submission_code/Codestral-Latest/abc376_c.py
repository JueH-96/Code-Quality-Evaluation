# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

n = int(data[0])
A = list(map(int, data[1:n+1]))
B = list(map(int, data[n+1:]))

# Sort the toy sizes and box sizes
A.sort()
B.sort()

# Initialize the minimum required box size
min_box_size = -1

# Check if it's possible to place all toys in the existing boxes
for i in range(n):
    if A[i] > B[i]:
        min_box_size = A[i]
        break

# If min_box_size is still -1, it means all toys can be placed in the existing boxes
if min_box_size == -1:
    print(-1)
else:
    # Check if purchasing a box of size min_box_size allows all toys to be placed
    for i in range(n):
        if A[i] > min_box_size:
            min_box_size = -1
            break

    print(min_box_size)