def solve():
    n = int(input())
    a = list(map(int, input().split()))

    max_product = 0
    for i in range(n):
        temp_a = a[:]
        temp_a[i] += 1
        product = 1
        for x in temp_a:
            product *= x
        max_product = max(max_product, product)

    print(max_product)

t = int(input())
for _ in range(t):
    solve()