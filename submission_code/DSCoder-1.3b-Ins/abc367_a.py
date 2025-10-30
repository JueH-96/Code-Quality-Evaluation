# YOUR CODE HERE
A, B, C = map(int, input().split())

if A < B and B < C or C < B and B < A:
    print("Yes")
else:
    print("No")