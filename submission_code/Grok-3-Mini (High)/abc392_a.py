# Read the three integers from standard input
a, b, c = map(int, input().split())

# Check if any two numbers multiply to the third
if (a * b == c) or (a * c == b) or (b * c == a):
    print("Yes")
else:
    print("No")