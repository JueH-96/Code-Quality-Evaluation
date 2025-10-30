# YOUR CODE HERE
# Read N and M
N, M = map(int, input().split())

# Read the sequence A
A = list(map(int, input().split()))

# Convert A to a set for O(1) lookup
A_set = set(A)

# Find all integers from 1 to N that are not in A
missing = []
for i in range(1, N + 1):
    if i not in A_set:
        missing.append(i)

# Output the count
print(len(missing))

# Output the missing integers if any
if missing:
    print(' '.join(map(str, missing)))