def solve():
    n, q = map(int, input().split())
    s = list(input())

    for _ in range(q):
        query = list(map(int, input().split()))
        query_type = query[0]

        if query_type == 1:
            l, r = query[1], query[2]
            for i in range(l - 1, r):
                s[i] = '1' if s[i] == '0' else '0'
        elif query_type == 2:
            l, r = query[1], query[2]
            sub = s[l - 1:r]
            is_good = True
            for i in range(len(sub) - 1):
                if sub[i] == sub[i + 1]:
                    is_good = False
                    break
            print("Yes" if is_good else "No")

solve()