A, B = map(int, input().split())
s = set()

if (A + B) % 2 == 0:
    s.add((A + B) // 2)
s.add(2 * A - B)
s.add(2 * B - A)

print(len(s))