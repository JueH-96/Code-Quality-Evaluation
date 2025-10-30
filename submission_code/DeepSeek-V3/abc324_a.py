# YOUR CODE HERE
n, *rest = map(int, open(0).read().split())
a = rest[:n]
if all(x == a[0] for x in a):
    print("Yes")
else:
    print("No")