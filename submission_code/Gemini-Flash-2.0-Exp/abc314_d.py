def solve():
    n = int(input())
    s = list(input())
    q = int(input())

    for _ in range(q):
        query = input().split()
        t = int(query[0])

        if t == 1:
            x = int(query[1])
            c = query[2]
            s[x - 1] = c
        elif t == 2:
            for i in range(n):
                if 'A' <= s[i] <= 'Z':
                    s[i] = s[i].lower()
        else:
            for i in range(n):
                if 'a' <= s[i] <= 'z':
                    s[i] = s[i].upper()

    print("".join(s))

solve()