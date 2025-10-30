n = int(input())
if n % 5 == 0:
    print(n)
else:
    lower = (n // 5) * 5
    higher = lower + 5
    if higher > 100:
        print(lower)
    else:
        print(higher if (higher - n) < (n - lower) else lower)