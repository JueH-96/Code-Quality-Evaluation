n = int(input())
lower = (n // 5) * 5
upper = lower + 5
if n - lower < upper - n:
    print(lower)
else:
    print(upper)