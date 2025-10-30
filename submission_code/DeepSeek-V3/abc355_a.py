# Read input values for A and B
A, B = map(int, input().split())

# Determine the possible culprits based on the witnesses' memories
# The culprit is not A and not B
possible_culprits = set([1, 2, 3]) - set([A, B])

# If there is only one possible culprit, print it
if len(possible_culprits) == 1:
    print(possible_culprits.pop())
else:
    # Otherwise, print -1
    print(-1)