t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    prod_mod = 1
    for num in a:
        prod_mod = (prod_mod * (num % k)) % k
    if prod_mod == 0:
        print(0)
    else:
        min_steps = float('inf')
        for num in a:
            rem = num % k
            steps = (k - rem) if rem != 0 else 0
            if steps < min_steps:
                min_steps = steps
        print(min_steps)