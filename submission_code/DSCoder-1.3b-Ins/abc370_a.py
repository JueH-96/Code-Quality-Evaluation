# YOUR CODE HERE
L, R = map(int, input().split())

if L == 1 and R == 1:
    print("Invalid")
elif L == 1:
    print("Yes")
else:
    print("No")