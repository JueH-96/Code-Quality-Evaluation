# YOUR CODE HERE
A, B = map(int, input().split())

# Set of all suspects
suspects = {1, 2, 3}

# Remove the suspects that are not the culprit according to the witnesses
suspects.discard(A)
suspects.discard(B)

# Check if there is exactly one suspect left
if len(suspects) == 1:
    print(suspects.pop())
else:
    print(-1)