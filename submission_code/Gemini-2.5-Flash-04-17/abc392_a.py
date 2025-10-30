# YOUR CODE HERE
# Read the three integers from standard input
# input() reads a line from stdin
# split() splits the line into parts by whitespace
# map(int, ...) converts each part to an integer
a, b, c = map(int, input().split())

# We need to check if there's a permutation (B_1, B_2, B_3)
# of (a, b, c) such that B_1 * B_2 = B_3.
# The set of values {B_1, B_2, B_3} is the same as {a, b, c}.
# The condition B_1 * B_2 = B_3 means that one element in the set {a, b, c}
# is the product of the other two elements.
# There are three unique pairs we can multiply: (a, b), (a, c), (b, c).
# We check if the product of any pair equals the third element.

if a * b == c or a * c == b or b * c == a:
    # If a*b equals c, the permutation could be (a, b, c) or (b, a, c).
    # If a*c equals b, the permutation could be (a, c, b) or (c, a, b).
    # If b*c equals a, the permutation could be (b, c, a) or (c, b, a).
    # If any of these conditions is true, it means a suitable permutation exists.
    print("Yes")
else:
    # If none of the conditions are true, no such permutation exists.
    print("No")