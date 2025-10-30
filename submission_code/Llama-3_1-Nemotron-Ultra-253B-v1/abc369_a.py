a, b = map(int, input().split())
s = set()
x1 = 2 * b - a
x2 = 2 * a - b
s.add(x1)
s.add(x2)
if (a + b) % 2 == 0:
    x3 = (a + b) // 2
    s.add(x3)
print(len(s))