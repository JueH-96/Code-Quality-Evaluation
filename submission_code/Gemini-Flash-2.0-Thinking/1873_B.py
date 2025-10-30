def solve():
    n = int(input())
    a = list(map(int, input().split()))

    max_product = 0

    for i in range(n):
        temp_a = list(a)
        temp_a[i] += 1
        current_product = 1
        for digit in temp_a:
            current_product *= digit
        max_product = max(max_product, current_product)

    print(max_product)

t = int(input())
for _ in range(t):
    solve()