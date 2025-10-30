x_a, y_a = map(int, input().split())
x_b, y_b = map(int, input().split())
x_c, y_c = map(int, input().split())

dot_ab_ac = (x_b - x_a) * (x_c - x_a) + (y_b - y_a) * (y_c - y_a)
dot_ba_bc = (x_a - x_b) * (x_c - x_b) + (y_a - y_b) * (y_c - y_b)
dot_ca_cb = (x_a - x_c) * (x_b - x_c) + (y_a - y_c) * (y_b - y_c)

if dot_ab_ac == 0 or dot_ba_bc == 0 or dot_ca_cb == 0:
    print("Yes")
else:
    print("No")