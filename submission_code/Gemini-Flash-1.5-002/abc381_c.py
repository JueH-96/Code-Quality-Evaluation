def solve():
    n = int(input())
    s = input()
    max_len = 0
    for i in range(n):
        for j in range(i, n):
            sub = s[i:j+1]
            if len(sub) % 2 == 0:
                continue
            mid = (len(sub) + 1) // 2
            is_11_22 = True
            if sub[mid - 1] != '/':
                is_11_22 = False
            else:
                for k in range(mid - 1):
                    if sub[k] != '1':
                        is_11_22 = False
                        break
                for k in range(mid, len(sub)):
                    if sub[k] != '2':
                        is_11_22 = False
                        break
            if is_11_22:
                max_len = max(max_len, len(sub))
    print(max_len)

solve()