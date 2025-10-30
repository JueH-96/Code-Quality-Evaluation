# YOUR CODE HERE

X, Y = map(int, input().split())

if X - Y <= 2 or Y - X <= 3:
    print("Yes")
else:
    print("No")