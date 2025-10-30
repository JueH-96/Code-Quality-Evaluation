# YOUR CODE HERE
A, B = map(int, input().split())

possible_x = set()

# x = 2B - A
possible_x.add(2 * B - A)

# x = 2A - B
possible_x.add(2 * A - B)

# x = (A + B) / 2 if A + B is even
if (A + B) % 2 == 0:
    possible_x.add((A + B) // 2)

print(len(possible_x))