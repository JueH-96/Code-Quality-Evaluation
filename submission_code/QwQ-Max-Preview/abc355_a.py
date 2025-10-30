# Read input values
A, B = map(int, input().split())

# Determine possible culprits by excluding A and B
possible = [x for x in [1, 2, 3] if x != A and x != B]

# Check if there's exactly one possible culprit
if len(possible) == 1:
    print(possible[0])
else:
    print(-1)