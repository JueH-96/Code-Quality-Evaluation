A, B = map(int, input().split())

possible_x = set()

# Case 1: x = 2B - A
possible_x.add(2 * B - A)

# Case 2: x = (A + B)/2 (only if A + B is even)
if (A + B) % 2 == 0:
    possible_x.add((A + B) // 2)

# Case 3: x = 2A - B
possible_x.add(2 * A - B)

print(len(possible_x))