def solve():
    n = int(input())
    s = input()
    q = int(input())
    s_list = list(s)

    for _ in range(q):
        c, d = input().split()
        for i in range(len(s_list)):
            if s_list[i] == c:
                s_list[i] = d

    print("".join(s_list))

solve()