A, B = map(int, input().split())

x_values = set()

# Case 1: x = 2B - A
x1 = 2 * B - A
x_values.add(x1)

# Case 2: x = 2A - B
x2 = 2 * A - B
x_values.add(x2)

# Case 3: x = (A + B) / 2, if it's an integer
if (A + B) % 2 == 0:
    x3 = (A + B) // 2
    x_values.add(x3)

print(len(x_values))