A, B = map(int, input().split())
possible = set()

if (A + B) % 2 == 0:
    possible.add((A + B) // 2)

possible.add(2 * A - B)
possible.add(2 * B - A)

print(len(possible))