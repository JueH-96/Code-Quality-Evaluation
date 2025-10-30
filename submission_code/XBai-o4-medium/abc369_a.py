A, B = map(int, input().split())

possible = set()
possible.add(2 * A - B)
possible.add(2 * B - A)

if (A + B) % 2 == 0:
    possible.add((A + B) // 2)

print(len(possible))