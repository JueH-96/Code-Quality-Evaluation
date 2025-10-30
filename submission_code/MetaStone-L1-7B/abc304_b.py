import math

N = int(input())

if N == 0:
    print(0)
else:
    k = math.floor(math.log10(N))
    t = k - 2
    if t <= 0:
        print(N)
    else:
        divisor = 10 ** t
        res = (N // divisor) * divisor
        print(res)