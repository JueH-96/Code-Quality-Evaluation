# Read input from stdin
A, B = map(int, input().split())

# Check if A and B are different
if A != B:
    # Calculate the culprit using the sum trick
    culprit = 6 - A - B
    print(culprit)
else:
    # If A and B are the same, cannot uniquely identify, print -1
    print(-1)