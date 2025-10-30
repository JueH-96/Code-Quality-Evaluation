n = int(input())
d = len(str(n))
if d <= 3:
    print(n)
else:
    divisor = 10 ** (d - 3)
    print((n // divisor) * divisor)