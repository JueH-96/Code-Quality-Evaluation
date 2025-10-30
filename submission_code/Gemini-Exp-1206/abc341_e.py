def solve():
    n, q = map(int, input().split())
    s = list(input())

    for _ in range(q):
        query = list(map(int, input().split()))
        type_ = query[0]
        l = query[1] - 1
        r = query[2] - 1

        if type_ == 1:
            for i in range(l, r + 1):
                s[i] = '1' if s[i] == '0' else '0'
        else:
            sub_s = s[l:r+1]
            is_good = True
            for i in range(len(sub_s) - 1):
                if sub_s[i] == sub_s[i+1]:
                    is_good = False
                    break
            print("Yes" if is_good else "No")

solve()