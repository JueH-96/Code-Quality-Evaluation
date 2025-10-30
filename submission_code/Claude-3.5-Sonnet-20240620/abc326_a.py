# YOUR CODE HERE
x, y = map(int, input().split())

difference = y - x

if abs(difference) <= 2 and difference > 0:
    print("Yes")
elif abs(difference) <= 3 and difference < 0:
    print("Yes")
else:
    print("No")