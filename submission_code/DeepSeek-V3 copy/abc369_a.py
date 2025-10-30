# YOUR CODE HERE
A, B = map(int, input().split())
unique_x = set()

# Case 1: x is the first element
# Sequence: x, A, B
# A - x = B - A => x = 2*A - B
x1 = 2 * A - B
unique_x.add(x1)

# Case 2: x is the second element
# Sequence: A, x, B
# x - A = B - x => x = (A + B) / 2
x2 = (A + B) / 2
if x2 == int(x2):
    unique_x.add(int(x2))

# Case 3: x is the third element
# Sequence: A, B, x
# B - A = x - B => x = 2*B - A
x3 = 2 * B - A
unique_x.add(x3)

# Case 4: x is the first element
# Sequence: x, B, A
# B - x = A - B => x = 2*B - A
x4 = 2 * B - A
unique_x.add(x4)

# Case 5: x is the second element
# Sequence: B, x, A
# x - B = A - x => x = (A + B) / 2
x5 = (A + B) / 2
if x5 == int(x5):
    unique_x.add(int(x5))

# Case 6: x is the third element
# Sequence: B, A, x
# A - B = x - A => x = 2*A - B
x6 = 2 * A - B
unique_x.add(x6)

print(len(unique_x))