n = int(input())
if n == 0:
    print(0)
else:
    digits = len(str(n))
    if digits <= 3:
        print(n)
    else:
        divisor = 10 ** (digits - 3)
        print((n // divisor) * divisor)