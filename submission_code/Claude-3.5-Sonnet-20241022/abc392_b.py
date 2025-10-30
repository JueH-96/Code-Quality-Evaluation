# Read input
N, M = map(int, input().split())
A = set(map(int, input().split()))

# Find missing numbers
missing = []
for i in range(1, N + 1):
    if i not in A:
        missing.append(i)

# Output results
print(len(missing))
if missing:  # Only print second line if there are missing numbers
    print(*missing)