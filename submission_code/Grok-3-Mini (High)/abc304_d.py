import sys
import bisect
import collections

# Read all input data
data = sys.stdin.read().split()
index = 0

# Read W and H
W = int(data[index])
index += 1
H = int(data[index])
index += 1

# Read N
N = int(data[index])
index += 1

# Read strawberries coordinates
strawberries = []
for _ in range(N):
    p = int(data[index])
    index += 1
    q = int(data[index])
    index += 1
    strawberries.append((p, q))

# Read A
A = int(data[index])
index += 1

# Read vertical cut lines
a_list = []
for _ in range(A):
    a_val = int(data[index])
    index += 1
    a_list.append(a_val)

# Read B
B = int(data[index])
index += 1

# Read horizontal cut lines
b_list = []
for _ in range(B):
    b_val = int(data[index])
    index += 1
    b_list.append(b_val)

# Build cut lines including boundaries
x_cuts = [0] + a_list + [W]
y_cuts = [0] + b_list + [H]

# Compute cell indices for each strawberry
cells = []
for p, q in strawberries:
    x_idx = bisect.bisect_right(x_cuts, p) - 1
    y_idx = bisect.bisect_right(y_cuts, q) - 1
    cells.append((x_idx, y_idx))

# Count the number of strawberries in each cell
counter = collections.Counter(cells)

# Find the maximum count
max_count = max(counter.values())

# Find the number of occupied cells and total cells
num_occupied = len(counter)
total_cells = (A + 1) * (B + 1)

# Determine the minimum count
if num_occupied < total_cells:
    min_count = 0
else:
    min_count = min(counter.values())

# Output the result
print(min_count, max_count)