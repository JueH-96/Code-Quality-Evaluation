def solve():
    a, b, c = map(int, input().split())
    s = a + b + c

    possible = False

    # Check for two groups
    if s % 2 == 0:
        target = s // 2
        if (a == target or
            b == target or
            c == target or
            a + b == target or
            a + c == target or
            b + c == target):
            possible = True

    # Check for three groups
    if s % 3 == 0:
        target = s // 3
        if a == target and b == target and c == target:
            possible = True

    if possible:
        print("Yes")
    else:
        print("No")

solve()