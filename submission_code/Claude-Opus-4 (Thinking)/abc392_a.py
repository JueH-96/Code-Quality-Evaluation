# YOUR CODE HERE
a1, a2, a3 = map(int, input().split())

if a1 * a2 == a3 or a1 * a3 == a2 or a2 * a3 == a1:
    print("Yes")
else:
    print("No")