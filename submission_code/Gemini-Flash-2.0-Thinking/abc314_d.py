def solve():
    n = int(input())
    s = input()
    q = int(input())
    s_list = list(s)

    for _ in range(q):
        query = input().split()
        t = int(query[0])

        if t == 1:
            x = int(query[1])
            c = query[2]
            s_list[x - 1] = c
        elif t == 2:
            s_list = [char.lower() for char in s_list]
        elif t == 3:
            s_list = [char.upper() for char in s_list]

    print("".join(s_list))

solve()