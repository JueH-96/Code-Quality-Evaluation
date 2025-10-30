x_a, y_a = map(int, input().split())
x_b, y_b = map(int, input().split())
x_c, y_c = map(int, input().split())

ab_sq = (x_b - x_a)**2 + (y_b - y_a)**2
ac_sq = (x_c - x_a)**2 + (y_c - y_a)**2
bc_sq = (x_c - x_b)**2 + (y_c - y_b)**2

if ab_sq + ac_sq == bc_sq or ab_sq + bc_sq == ac_sq or ac_sq + bc_sq == ab_sq:
    print("Yes")
else:
    print("No")