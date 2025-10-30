# Read inputs
line1 = input()
line2 = input()
line3 = input()

# Split and convert to integers
x_A, y_A = map(int, line1.split())
x_B, y_B = map(int, line2.split())
x_C, y_C = map(int, line3.split())

def squared_distance(x1, y1, x2, y2):
    return (x2 - x1) ** 2 + (y2 - y1) ** 2

# Calculate squared distances
a = squared_distance(x_B, y_B, x_C, y_C)
b = squared_distance(x_A, y_A, x_C, y_C)
c = squared_distance(x_A, y_A, x_B, y_B)

# Check Pythagorean theorem
if a + b == c or a + c == b or b + c == a:
    print("Yes")
else:
    print("No")