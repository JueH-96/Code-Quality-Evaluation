# YOUR CODE HERE
A, B = map(int, input().split())

# Determine the culprit based on the memories of Ringo and Snuke
suspects = {1, 2, 3}
if A in suspects:
    suspects.remove(A)
if B in suspects:
    suspects.remove(B)

# Check if there is exactly one suspect left
if len(suspects) == 1:
    print(suspects.pop())
else:
    print(-1)