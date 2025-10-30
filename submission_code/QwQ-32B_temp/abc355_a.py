# Read input values
A, B = map(int, input().split())

# Determine possible culprits
possible = []
for person in [1, 2, 3]:
    if person != A and person != B:
        possible.append(person)

# Check if there's exactly one possible culprit
if len(possible) == 1:
    print(possible[0])
else:
    print(-1)