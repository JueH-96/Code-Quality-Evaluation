n = int(input())
if n < 1000:
    print(n)
else:
    divisor = 10  # Default, will be updated
    if n >= 10**8:
        divisor = 10**6
    elif n >= 10**7:
        divisor = 10**5
    elif n >= 10**6:
        divisor = 10**4
    elif n >= 10**5:
        divisor = 10**3
    elif n >= 10**4:
        divisor = 10**2
    elif n >= 10**3:
        divisor = 10**1
    res = (n // divisor) * divisor
    print(res)