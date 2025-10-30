t = int(input().strip())
for _ in range(t):
    n = int(input().strip())
    a = list(map(int, input().strip().split()))
    if 0 in a:
        print(0)
    else:
        min_val = min(a)
        if min_val <= 1:
            a[a.index(min_val)] += 1
        else:
            a[a.index(min_val)] = min_val - 1
        product = 1
        for i in a:
            product *= i
        print(product)