n = int(input())

if n <= 10:
    print(n - 1)
else:
    n -= 10
    if n <= 9:
        print(n * 11)
    else:
        n -= 9
        l = 3
        while True:
            m = (l + 1) // 2
            count = 9 * (10 ** (m - 1))
            if n > count:
                n -= count
                l += 1
            else:
                break
        m = (l + 1) // 2
        first_part = 10 ** (m - 1) + (n - 1)
        s = str(first_part)
        if l % 2 == 0:
            res = s + s[::-1]
        else:
            res = s + s[:-1][::-1]
        print(res)