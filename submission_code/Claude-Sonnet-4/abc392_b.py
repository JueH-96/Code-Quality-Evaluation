# YOUR CODE HERE
# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Convert A to a set for faster lookup
A_set = set(A)

# Find missing numbers
missing = []
for i in range(1, N + 1):
    if i not in A_set:
        missing.append(i)

# Output
print(len(missing))
if missing:
    print(*missing)