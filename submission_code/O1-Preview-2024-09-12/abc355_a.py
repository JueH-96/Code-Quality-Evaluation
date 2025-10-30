# YOUR CODE HERE
A, B = map(int, input().split())
culprits = {1, 2, 3} - {A, B}
if len(culprits) == 1:
    print(culprits.pop())
else:
    print(-1)