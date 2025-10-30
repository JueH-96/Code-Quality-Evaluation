# YOUR CODE HERE
a, b, c = map(int, input().split())

if b < c:
    if b < a < c:
        print("Yes")
    else:
        print("No")
else:
    if (b < a < 24) or (0 <= a < c):
        print("Yes")
    else:
        print("No")