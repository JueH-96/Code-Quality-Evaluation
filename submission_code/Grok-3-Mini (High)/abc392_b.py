import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()

# Convert N and M from strings to integers
N = int(data[0])
M = int(data[1])

# Convert the list of A elements from strings to integers
A = list(map(int, data[2:2+M]))

# Create a set from A for fast lookup
A_set = set(A)

# Find all numbers from 1 to N that are not in A_set
missing = [num for num in range(1, N+1) if num not in A_set]

# Output the count of missing numbers
print(len(missing))

# Output the missing numbers separated by spaces, or an empty line if none
print(' '.join(map(str, missing)))