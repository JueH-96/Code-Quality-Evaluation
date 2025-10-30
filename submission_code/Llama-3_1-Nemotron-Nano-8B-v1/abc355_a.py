# Read the input values
A, B = map(int, input().split())

# Check if A and B are different
if A != B:
    # The culprit is the third person not mentioned by either
    print(6 - A - B)
else:
    # Both witnesses say the same person, so two possibilities remain
    print(-1)