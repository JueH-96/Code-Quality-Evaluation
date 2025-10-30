# Read the input integers A and B
A, B = map(int, input().split())

# Use a set to store the distinct possible values of x
possible_x_values = set()

# Case 1: x is the middle term.
# The sequence could be (A, x, B) or (B, x, A).
# For an arithmetic sequence, 2*x = A + B.
# So, x = (A + B) / 2.
# For x to be an integer, (A + B) must be even.
if (A + B) % 2 == 0:
    x1 = (A + B) // 2
    possible_x_values.add(x1)

# Case 2: A is the middle term.
# The sequence could be (x, A, B) or (B, A, x).
# For an arithmetic sequence, 2*A = x + B.
# So, x = 2*A - B.
x2 = 2 * A - B
possible_x_values.add(x2)

# Case 3: B is the middle term.
# The sequence could be (x, B, A) or (A, B, x).
# For an arithmetic sequence, 2*B = x + A.
# So, x = 2*B - A.
x3 = 2 * B - A
possible_x_values.add(x3)

# The number of distinct integers x is the size of the set.
print(len(possible_x_values))