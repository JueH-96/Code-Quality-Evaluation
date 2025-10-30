# YOUR CODE HERE
A, B, C = map(int, input().split())

if B < C:
    if B <= A < C:
        print("Yes")
    else:
        print("No")
else:
    if A >= B or A < C:
        print("Yes")
    else:
        print("No")