# YOUR CODE HERE
X, Y = map(int, input().split())
difference = Y - X
if difference > 0:
    if difference <= 2:
        print("Yes")
    else:
        print("No")
else:
    if difference >= -3:
        print("Yes")
    else:
        print("No")