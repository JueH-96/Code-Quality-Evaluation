# YOUR CODE HERE
l, r = map(int, input().split())
total = l + r
if total == 1:
    print("Yes" if l == 1 else "No")
else:
    print("Invalid")