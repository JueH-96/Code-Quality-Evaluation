A, B = map(int, input().split())

# Set to store unique values of x
valid_x = set()

# Case 1: x = 2B - A
valid_x.add(2 * B - A)

# Case 2: x = 2A - B
valid_x.add(2 * A - B)

# Case 3: x = (A + B) / 2 (only if A + B is even)
if (A + B) % 2 == 0:
    valid_x.add((A + B) // 2)

print(len(valid_x))