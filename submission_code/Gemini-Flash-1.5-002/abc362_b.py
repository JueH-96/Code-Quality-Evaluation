# YOUR CODE HERE
def dist_sq(x1, y1, x2, y2):
    return (x1 - x2)**2 + (y1 - y2)**2

x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

a2 = dist_sq(x_B, y_B, x_C, y_C)
b2 = dist_sq(x_A, y_A, x_C, y_C)
c2 = dist_sq(x_A, y_A, x_B, y_B)

if a2 + b2 == c2 or a2 + c2 == b2 or b2 + c2 == a2:
    print("Yes")
else:
    print("No")