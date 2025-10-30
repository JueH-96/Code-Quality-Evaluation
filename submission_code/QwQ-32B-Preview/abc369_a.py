A, B = map(int, input().split())

x_values = set()

x1 = 2 * B - A
x_values.add(x1)

if (A + B) % 2 == 0:
    x2 = (A + B) // 2
    x_values.add(x2)

x3 = 2 * A - B
x_values.add(x3)

print(len(x_values))