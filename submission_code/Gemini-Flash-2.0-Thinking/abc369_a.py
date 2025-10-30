def solve():
    a, b = map(int, input().split())

    possible_x = set()

    # Case 1: A, B, x or x, B, A
    x1 = 2 * b - a
    possible_x.add(x1)

    # Case 2: A, x, B or B, x, A
    if (a + b) % 2 == 0:
        x2 = (a + b) // 2
        possible_x.add(x2)

    # Case 3: x, A, B or B, A, x
    x3 = 2 * a - b
    possible_x.add(x3)

    print(len(possible_x))

solve()