n = int(input())
if n == 0:
    print(0)
else:
    k = len(str(n))
    if k < 4:
        print(n)
    else:
        m = k - 1
        d = m - 2
        divisor = 10 ** d
        approx = (n // divisor) * divisor
        print(approx)