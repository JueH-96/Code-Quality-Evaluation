A, B = map(int, input().split())
x1 = 2 * A - B
x2 = 2 * B - A
s = {x1, x2}
if (A + B) % 2 == 0:
    x3 = (A + B) // 2
    s.add(x3)
print(len(s))