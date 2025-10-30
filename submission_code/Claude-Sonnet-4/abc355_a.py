# YOUR CODE HERE
A, B = map(int, input().split())

# Create set of all suspects
suspects = {1, 2, 3}

# Remove the people who are not culprits according to witnesses
suspects.discard(A)
suspects.discard(B)

# If exactly one person remains, they are the culprit
if len(suspects) == 1:
    print(suspects.pop())
else:
    print(-1)