n, x, y = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

sum_a = sum(a)
sum_b = sum(b)

if sum_a <= x and sum_b <= y:
    print(n)
else:
    # Compute k_a
    sorted_a = sorted(a, reverse=True)
    prefix = 0
    ka = float('inf')
    for i in range(n):
        prefix += sorted_a[i]
        if prefix > x:
            ka = i + 1
            break
    
    # Compute k_b
    sorted_b = sorted(b, reverse=True)
    prefix = 0
    kb = float('inf')
    for i in range(n):
        prefix += sorted_b[i]
        if prefix > y:
            kb = i + 1
            break
    
    print(min(ka, kb))