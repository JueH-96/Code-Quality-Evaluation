A, M, L, R = map(int, input().split())

rem = A % M
rem_L = L % M
delta = (rem - rem_L) % M
x1 = L + delta

if x1 > R:
    print(0)
else:
    rem_R = R % M
    delta = (rem_R - rem) % M
    x2 = R - delta
    if x2 < L:
        print(0)
    else:
        if x1 > x2:
            print(0)
        else:
            count = (x2 - x1) // M + 1
            print(count)