A, B = map(int, input().split())
s = set()
x2 = 2 * A - B
x3 = 2 * B - A
s.add(x2)
s.add(x3)
if (A + B) % 2 == 0:
    s.add((A + B) // 2)
print(len(s))