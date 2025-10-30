# YOUR CODE HERE

# Read the input
A, B = map(int, input().split())

# Check if the culprit can be uniquely identified
if A == B:
    print(-1)
else:
    print(A)