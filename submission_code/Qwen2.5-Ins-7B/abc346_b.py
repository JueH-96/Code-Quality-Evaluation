# YOUR CODE HERE
w, b = map(int, input().split())
if w == 0 and b == 0:
    print("No")
elif w == 0:
    print("No" if b % 7 else "Yes")
elif b == 0:
    print("No" if w % 5 else "Yes")
else:
    print("Yes")