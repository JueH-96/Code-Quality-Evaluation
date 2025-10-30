t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    max_prod = -1
    for i in range(n):
        temp = a.copy()
        temp[i] += 1
        product = 1
        for num in temp:
            product *= num
        if product > max_prod:
            max_prod = product
    print(max_prod)