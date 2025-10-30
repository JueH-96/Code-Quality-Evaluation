# YOUR CODE HERE
A, B, C = map(int, input().split())

if A < B:
    if B < C:
        print("Yes")
    elif A < C:
        print("Yes")
    else:
        print("No")
else:
    if B < C and C < 24:
        print("Yes")
    elif A < 24:
        print("Yes")
    else:
        print("No")