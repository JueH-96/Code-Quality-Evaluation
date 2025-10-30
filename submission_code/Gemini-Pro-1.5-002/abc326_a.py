# YOUR CODE HERE
x, y = map(int, input().split())
diff = y - x
if diff >= -3 and diff <= 2:
    print("Yes")
else:
    print("No")