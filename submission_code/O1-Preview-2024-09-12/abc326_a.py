# YOUR CODE HERE
X, Y = map(int, input().split())
delta = Y - X

if delta > 0:
    if delta <= 2:
        print("Yes")
    else:
        print("No")
elif delta < 0:
    if abs(delta) <= 3:
        print("Yes")
    else:
        print("No")