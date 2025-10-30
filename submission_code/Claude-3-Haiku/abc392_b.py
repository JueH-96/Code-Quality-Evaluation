import sys

# Read input
line1 = input().split()
N, M = map(int, line1)
A = list(map(int, input().split()))

# Find missing numbers
missing = [i for i in range(1, N+1) if i not in A]

# Print output
print(len(missing))
if missing:
    print(" ".join(map(str, missing)))