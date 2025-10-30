# Read input values for A and B
A, B = map(int, input().split())

# Determine the possible culprits based on the witnesses' memories
# The culprit is not A and not B
possible_culprits = {1, 2, 3} - {A, B}

# Check if there is exactly one possible culprit
if len(possible_culprits) == 1:
    print(possible_culprits.pop())
else:
    print(-1)