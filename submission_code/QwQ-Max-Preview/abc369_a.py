a, b = map(int, input().split())

possible = set()

x2 = 2 * a - b
x3 = 2 * b - a

possible.add(x2)
possible.add(x3)

if (a + b) % 2 == 0:
    x1 = (a + b) // 2
    possible.add(x1)

print(len(possible))