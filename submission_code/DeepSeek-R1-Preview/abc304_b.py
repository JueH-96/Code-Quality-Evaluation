n = int(input())

if n == 0:
    print(0)
else:
    thresholds = [
        (10**8, 10**6),
        (10**7, 10**5),
        (10**6, 10**4),
        (10**5, 10**3),
        (10**4, 10**2),
        (10**3, 10),
        (0, 1)
    ]
    for lower, divisor in thresholds:
        if n >= lower:
            print((n // divisor) * divisor)
            break