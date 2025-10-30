N = int(input())

if N % 5 == 0:
    nearest = N
else:
    q = N // 5
    lower = q * 5
    higher = (q + 1) * 5
    if higher > 100:
        nearest = lower
    else:
        if N - lower < higher - N:
            nearest = lower
        else:
            nearest = higher

print(nearest)