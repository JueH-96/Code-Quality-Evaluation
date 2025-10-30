# YOUR CODE HERE
r, g, b = map(int, input().split())
c = input()

if c == "Red":
    print(min(g, b))
elif c == "Green":
    print(min(r, b))
elif c == "Blue":
    print(min(r, g))