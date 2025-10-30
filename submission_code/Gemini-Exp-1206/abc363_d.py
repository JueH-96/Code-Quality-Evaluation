def solve():
    n = int(input())
    if n == 1:
        print(0)
        return

    n -= 1
    length = 1
    count = 9
    while n > count:
        n -= count
        length += 1
        if length % 2 == 1:
            count = 9 * (10 ** (length // 2))
        else:
            count = 9 * (10 ** (length // 2 - 1))

    if length % 2 == 1:
        half = 10 ** (length // 2) + n - 1
        s = str(half)
        print(s + s[-2::-1])
    else:
        half = 10 ** (length // 2 - 1) + n - 1
        s = str(half)
        print(s + s[::-1])

solve()