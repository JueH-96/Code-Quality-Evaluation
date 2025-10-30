t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    max_prod = 0
    for i in range(n):
        new_a = a.copy()
        new_a[i] += 1
        product = 1
        for num in new_a:
            product *= num
        if product > max_prod:
            max_prod = product
    print(max_prod)