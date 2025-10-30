# Read input
A, B = map(int, input().split())

# Initialize set of possible culprits
culprits = {1, 2, 3}

# Remove A and B from the set
culprits.discard(A)
culprits.discard(B)

# Determine the output
if len(culprits) == 1:
    print(culprits.pop())
else:
    print(-1)