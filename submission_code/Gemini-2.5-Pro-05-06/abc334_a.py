# YOUR CODE HERE
B, G = map(int, input().split())

if B > G:
    print("Bat")
else:
    # Since B != G is a constraint, if B is not greater than G,
    # then G must be greater than B.
    print("Glove")