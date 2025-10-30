# YOUR CODE HERE
A, B = map(int, input().split())

# If both witnesses exclude the same person, we cannot determine the culprit.
if A == B:
    print(-1)
else:
    # The culprit is the person not excluded by either witness.
    # This can be found by summing 1, 2, 3 and subtracting A and B.
    culprit = 6 - A - B
    print(culprit)