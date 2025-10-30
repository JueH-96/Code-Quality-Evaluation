n = int(input())
if n <= 999:
    print(n)
else:
    if n < 10**4:
        divisor = 10**1
    elif n < 10**5:
        divisor = 10**2
    elif n < 10**6:
        divisor = 10**3
    elif n < 10**7:
        divisor = 10**4
    elif n < 10**8:
        divisor = 10**5
    else:
        divisor = 10**6
    print((n // divisor) * divisor)