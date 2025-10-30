# Read input
A, B = map(int, input().split())

# Find possible culprits who are not A and not B
culprits = [x for x in [1, 2, 3] if x != A and x != B]

# Output the result based on the number of culprits
if len(culprits) == 1:
    print(culprits[0])
else:
    print(-1)