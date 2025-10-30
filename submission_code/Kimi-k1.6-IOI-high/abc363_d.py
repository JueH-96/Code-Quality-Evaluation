n = int(input())
if n == 1:
    print(0)
else:
    idx = n - 1
    cumulative = 0
    d = 1
    while True:
        if d == 1:
            cnt = 10
        else:
            m = (d + 1) // 2
            cnt = 9 * 10 ** (m - 1)
        if cumulative + cnt > idx:
            break
        else:
            cumulative += cnt
            d += 1
    if d == 1:
        print(idx)
    else:
        m = (d + 1) // 2
        k = idx - cumulative
        root = 10 ** (m - 1) + k
        s = str(root)
        if d % 2 == 0:
            print(s + s[::-1])
        else:
            print(s + s[:-1][::-1])