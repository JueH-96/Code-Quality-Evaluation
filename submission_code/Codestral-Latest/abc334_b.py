# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

A = int(data[0])
M = int(data[1])
L = int(data[2])
R = int(data[3])

# Find the first Christmas tree to the east of L
first_east = ((L - A + M - 1) // M) * M + A

# Find the first Christmas tree to the west of R
first_west = ((R - A) // M) * M + A

# Calculate the number of Christmas trees between L and R
count = 0
if first_east <= R:
    count = (first_west - first_east) // M + 1

print(count)