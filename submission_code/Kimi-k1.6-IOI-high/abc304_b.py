n = int(input())
s = str(n)
d = len(s)
if d <= 3:
    print(n)
else:
    trunc_digits = d - 3
    divisor = 10 ** trunc_digits
    print((n // divisor) * divisor)