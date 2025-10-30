import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()

# Initialize index
idx = 0

# Read N and M
N = int(data[idx])
idx += 1
M = int(data[idx])
idx += 1

# Read and convert A to integers
A = list(map(int, data[idx:idx + N]))
idx += N

# Read and convert B to integers
B = list(map(int, data[idx:idx + M]))

# Create a set for fast lookup of elements in A
A_set = set(A)

# Combine and sort all elements
all_sorted = sorted(A + B)

# Check for consecutive elements both in A
found = False
for i in range(len(all_sorted) - 1):
    if all_sorted[i] in A_set and all_sorted[i + 1] in A_set:
        found = True
        break

# Output the result
if found:
    print("Yes")
else:
    print("No")