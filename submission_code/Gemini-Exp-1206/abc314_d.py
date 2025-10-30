def solve():
    n = int(input())
    s = list(input())
    q = int(input())

    for _ in range(q):
        t, x, c = input().split()
        t = int(t)
        x = int(x)

        if t == 1:
            s[x - 1] = c
        elif t == 2:
            s = [char.lower() for char in s]
        elif t == 3:
            s = [char.upper() for char in s]

    print("".join(s))

solve()