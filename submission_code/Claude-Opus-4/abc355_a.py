# YOUR CODE HERE
A, B = map(int, input().split())

if A == B:
    print(-1)
else:
    # Find the person who is neither A nor B
    suspects = {1, 2, 3}
    suspects.remove(A)
    suspects.remove(B)
    print(suspects.pop())