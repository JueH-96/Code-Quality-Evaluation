A, B = map(int, input().split())

x1 = 2 * A - B
x2 = 2 * B - A
sum_ab = A + B

x3 = None
if sum_ab % 2 == 0:
    x3 = sum_ab // 2

possible_x = {x1, x2}
if x3 is not None:
    possible_x.add(x3)

print(len(possible_x))